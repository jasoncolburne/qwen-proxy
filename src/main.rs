use std::env;
use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

struct AppState {
    tokenizer: Tokenizer,
    client: reqwest::Client,
    upstream: String,
    api_key: Option<String>,          // key we expect from clients
    upstream_api_key: Option<String>, // key we send to vMLX
}

#[derive(Deserialize)]
struct MessagesRequest {
    #[serde(default)]
    system: Option<SystemPrompt>,
    #[serde(default)]
    messages: Vec<Message>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: Content,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl Default for Content {
    fn default() -> Self {
        Content::Text(String::new())
    }
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Serialize)]
struct CountTokensResponse {
    input_tokens: usize,
}

fn extract_text(content: &Content) -> String {
    match content {
        Content::Text(s) => s.clone(),
        Content::Blocks(blocks) => blocks
            .iter()
            .filter(|b| b.r#type == "text")
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join(""),
    }
}

async fn count_tokens(
    State(state): State<Arc<AppState>>,
    axum::Json(body): axum::Json<MessagesRequest>,
) -> impl IntoResponse {
    let mut all_text = String::new();

    if let Some(system) = &body.system {
        match system {
            SystemPrompt::Text(s) => all_text.push_str(s),
            SystemPrompt::Blocks(blocks) => {
                for block in blocks {
                    if block.r#type == "text" {
                        if let Some(t) = &block.text {
                            all_text.push_str(t);
                        }
                    }
                }
            }
        }
    }

    for msg in &body.messages {
        all_text.push_str(&extract_text(&msg.content));
    }

    let encoding = state.tokenizer.encode(all_text, false);
    match encoding {
        Ok(enc) => axum::Json(CountTokensResponse {
            input_tokens: enc.get_ids().len(),
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Tokenizer error: {e}"),
        )
            .into_response(),
    }
}

fn check_auth(headers: &axum::http::HeaderMap, expected: &Option<String>) -> bool {
    match expected {
        None => true,
        Some(expected) => {
            let provided = headers
                .get("x-api-key")
                .or_else(|| headers.get("authorization"))
                .and_then(|v| v.to_str().ok())
                .map(|v| v.trim_start_matches("Bearer ").trim());
            matches!(provided, Some(key) if key == expected.as_str())
        }
    }
}

fn translate_user_content(blocks: &[Value], out: &mut Vec<Value>) {
    let mut pending_text = String::new();
    let flush_text = |buf: &mut String, out: &mut Vec<Value>| {
        if !buf.is_empty() {
            out.push(json!({"role": "user", "content": buf.clone()}));
            buf.clear();
        }
    };
    for block in blocks {
        let t = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match t {
            "text" => {
                if let Some(s) = block.get("text").and_then(|v| v.as_str()) {
                    if !pending_text.is_empty() {
                        pending_text.push('\n');
                    }
                    pending_text.push_str(s);
                }
            }
            "tool_result" => {
                flush_text(&mut pending_text, out);
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let content_text = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(arr)) => arr
                        .iter()
                        .filter_map(|b| {
                            if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                                b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    _ => String::new(),
                };
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content_text,
                }));
            }
            _ => {}
        }
    }
    flush_text(&mut pending_text, out);
}

fn translate_assistant_content(blocks: &[Value], out: &mut Vec<Value>) {
    let mut text_content = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for block in blocks {
        let t = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match t {
            "text" => {
                if let Some(s) = block.get("text").and_then(|v| v.as_str()) {
                    if !text_content.is_empty() {
                        text_content.push('\n');
                    }
                    text_content.push_str(s);
                }
            }
            "tool_use" => {
                let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = block
                    .get("input")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "{}".to_string());
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments}
                }));
            }
            _ => {}
        }
    }

    if tool_calls.is_empty() {
        out.push(json!({"role": "assistant", "content": text_content}));
    } else {
        let mut msg = serde_json::Map::new();
        msg.insert("role".to_string(), json!("assistant"));
        msg.insert(
            "content".to_string(),
            if text_content.is_empty() {
                Value::Null
            } else {
                Value::String(text_content)
            },
        );
        msg.insert("tool_calls".to_string(), Value::Array(tool_calls));
        out.push(Value::Object(msg));
    }
}

fn anthropic_to_openai(body: &Value) -> Value {
    let mut messages_out: Vec<Value> = Vec::new();

    if let Some(system) = body.get("system") {
        let text = match system {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        if !text.is_empty() {
            messages_out.push(json!({"role": "system", "content": text}));
        }
    }

    if let Some(messages) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            match msg.get("content") {
                Some(Value::String(s)) => {
                    messages_out.push(json!({"role": role, "content": s}));
                }
                Some(Value::Array(blocks)) => {
                    if role == "assistant" {
                        translate_assistant_content(blocks, &mut messages_out);
                    } else {
                        translate_user_content(blocks, &mut messages_out);
                    }
                }
                _ => {
                    messages_out.push(json!({"role": role, "content": ""}));
                }
            }
        }
    }

    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let stream_requested = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut out = json!({
        "model": model,
        "messages": messages_out,
        "stream": stream_requested,
    });
    if let Some(mt) = body.get("max_tokens") {
        let capped = mt.as_u64().map(|n| n.min(65536)).unwrap_or(65536);
        out["max_tokens"] = json!(capped);
    }
    if let Some(t) = body.get("temperature") {
        out["temperature"] = t.clone();
    }
    if let Some(tp) = body.get("top_p") {
        out["top_p"] = tp.clone();
    }
    if let Some(ss) = body.get("stop_sequences") {
        out["stop"] = ss.clone();
    }
    out
}

fn map_stop_reason(fr: &str) -> &'static str {
    match fr {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < buf.len() {
        if buf[i] == b'\n' && i + 1 < buf.len() && buf[i + 1] == b'\n' {
            return Some(i + 2);
        }
        if i + 3 < buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
            return Some(i + 4);
        }
        i += 1;
    }
    None
}

fn sse_event(event: &str, data: &Value) -> Bytes {
    Bytes::from(format!("event: {}\ndata: {}\n\n", event, data))
}

fn tokenize_len(tokenizer: &Tokenizer, text: &str) -> u64 {
    tokenizer
        .encode(text, false)
        .map(|enc| enc.get_ids().len() as u64)
        .unwrap_or(0)
}

fn collect_block_text(block: &Value, out: &mut String) {
    let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match block_type {
        "text" => {
            if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                out.push_str(t);
                out.push('\n');
            }
        }
        "tool_result" => {
            if let Some(name) = block.get("tool_use_id").and_then(|v| v.as_str()) {
                out.push_str(name);
                out.push('\n');
            }
            match block.get("content") {
                Some(Value::String(s)) => {
                    out.push_str(s);
                    out.push('\n');
                }
                Some(Value::Array(arr)) => {
                    for inner in arr {
                        collect_block_text(inner, out);
                    }
                }
                _ => {}
            }
        }
        "tool_use" => {
            if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                out.push_str(name);
                out.push('\n');
            }
            if let Some(input) = block.get("input") {
                out.push_str(&input.to_string());
                out.push('\n');
            }
        }
        _ => {}
    }
}

fn anthropic_full_text(body: &Value) -> String {
    let mut out = String::new();

    if let Some(system) = body.get("system") {
        match system {
            Value::String(s) => {
                out.push_str(s);
                out.push('\n');
            }
            Value::Array(arr) => {
                for block in arr {
                    collect_block_text(block, &mut out);
                }
            }
            _ => {}
        }
    }

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            match msg.get("content") {
                Some(Value::String(s)) => {
                    out.push_str(s);
                    out.push('\n');
                }
                Some(Value::Array(blocks)) => {
                    for block in blocks {
                        collect_block_text(block, &mut out);
                    }
                }
                _ => {}
            }
        }
    }

    out
}

enum Segment {
    Text(String),
    ToolCall { name: String, input: Value },
}

fn parse_tool_call(body: &str) -> Option<(String, Value)> {
    let body = body.trim();
    let (name, mut rest) = match body.find(char::is_whitespace) {
        Some(pos) => (body[..pos].to_string(), body[pos..].trim_start()),
        None => (body.to_string(), ""),
    };
    if name.is_empty() {
        return None;
    }

    let mut input = serde_json::Map::new();
    while !rest.is_empty() {
        let eq = match rest.find('=') {
            Some(p) => p,
            None => break,
        };
        let key = rest[..eq].trim().to_string();
        let after_eq = &rest[eq + 1..];
        if !after_eq.starts_with('"') {
            break;
        }
        let after_quote = &after_eq[1..];

        let mut val = String::new();
        let mut end_byte: Option<usize> = None;
        let mut chars = after_quote.char_indices();
        while let Some((i, c)) = chars.next() {
            if c == '\\' {
                if let Some((_, next)) = chars.next() {
                    match next {
                        'n' => val.push('\n'),
                        't' => val.push('\t'),
                        'r' => val.push('\r'),
                        '"' => val.push('"'),
                        '\\' => val.push('\\'),
                        other => {
                            val.push('\\');
                            val.push(other);
                        }
                    }
                }
            } else if c == '"' {
                end_byte = Some(i);
                break;
            } else {
                val.push(c);
            }
        }

        let end = match end_byte {
            Some(e) => e,
            None => break,
        };
        if !key.is_empty() {
            input.insert(key, Value::String(val));
        }
        rest = after_quote[end + 1..].trim_start();
    }

    Some((name, Value::Object(input)))
}

fn parse_segments(full: &str) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut remaining = full;
    while let Some(start) = remaining.find("<tool_call>") {
        let (before, rest) = remaining.split_at(start);
        if !before.is_empty() {
            out.push(Segment::Text(before.to_string()));
        }
        let after_open = &rest["<tool_call>".len()..];
        let (body, next) = match after_open.find("</tool_call>") {
            Some(end) => (
                &after_open[..end],
                &after_open[end + "</tool_call>".len()..],
            ),
            None => (after_open, ""),
        };
        if let Some((name, input)) = parse_tool_call(body) {
            out.push(Segment::ToolCall { name, input });
        }
        remaining = next;
    }
    if !remaining.is_empty() {
        out.push(Segment::Text(remaining.to_string()));
    }
    out
}

async fn messages(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();

    let path = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    println!("[messages] {} {}", &method, path);
    println!("[messages] request headers:");
    for (k, v) in headers.iter() {
        println!("  {}: {}", k, v.to_str().unwrap_or("<binary>"));
    }

    if !check_auth(&headers, &state.api_key) {
        println!("[messages] unauthorized request to {}", path);
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }

    let body_bytes = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("Body read error: {e}")).into_response();
        }
    };

    println!(
        "[messages] request body ({} bytes): {}",
        body_bytes.len(),
        String::from_utf8_lossy(&body_bytes)
            .chars()
            .take(500)
            .collect::<String>()
    );

    let anthropic_req: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            println!("[messages] JSON parse error: {e}");
            return (StatusCode::BAD_REQUEST, format!("JSON parse error: {e}")).into_response();
        }
    };

    let model = anthropic_req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let stream_requested = anthropic_req
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let openai_req = anthropic_to_openai(&anthropic_req);

    let input_text = anthropic_full_text(&anthropic_req);
    let input_tokens = tokenize_len(&state.tokenizer, &input_text);
    println!(
        "[messages] computed input_tokens: {} (from {} chars)",
        input_tokens,
        input_text.len()
    );

    let url = format!("{}/v1/chat/completions", state.upstream);
    println!(
        "[messages] translating -> {} (model={}, stream={})",
        url, model, stream_requested
    );
    println!(
        "[messages] translated messages: {}",
        serde_json::to_string(openai_req.get("messages").unwrap_or(&Value::Null))
            .unwrap_or_else(|_| "<unserializable>".to_string())
    );

    let mut upstream_req = state.client.post(&url).json(&openai_req);
    for (key, value) in headers.iter() {
        let k = key.as_str();
        if k == "host"
            || k == "content-length"
            || k == "content-type"
            || k == "x-api-key"
            || k == "authorization"
            || k == "anthropic-beta"
            || k == "anthropic-version"
        {
            continue;
        }
        upstream_req = upstream_req.header(key, value);
    }
    if let Some(upstream_key) = &state.upstream_api_key {
        upstream_req = upstream_req.header("authorization", format!("Bearer {}", upstream_key));
    }

    println!("[messages] sending to upstream: {}", url);
    let resp = match upstream_req.send().await {
        Ok(r) => r,
        Err(e) => {
            println!("[messages] upstream request failed: {e}");
            return (StatusCode::BAD_GATEWAY, format!("Upstream error: {e}")).into_response();
        }
    };

    let status = resp.status();
    println!("[messages] upstream status: {}", status);
    println!("[messages] upstream response headers:");
    for (k, v) in resp.headers().iter() {
        println!("  {}: {}", k, v.to_str().unwrap_or("<binary>"));
    }
    if !status.is_success() {
        let code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let bytes = resp.bytes().await.unwrap_or_default();
        println!(
            "[messages] upstream non-success {}: {}",
            status,
            String::from_utf8_lossy(&bytes)
                .chars()
                .take(500)
                .collect::<String>()
        );
        return (code, bytes).into_response();
    }

    if stream_requested {
        println!("[messages] piping translated SSE stream");
        let msg_id = format!(
            "msg_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let model_for_stream = model.clone();
        let state_for_stream = state.clone();
        let input_tokens_start = input_tokens;
        let mut upstream_stream = resp.bytes_stream();

        let translated = async_stream::stream! {
            let mut emit_count: u64 = 0;

            macro_rules! emit {
                ($event:expr, $data:expr) => {{
                    let bytes = sse_event($event, $data);
                    emit_count += 1;
                    println!(
                        "[sse-emit] {} {}",
                        $event,
                        String::from_utf8_lossy(&bytes)
                            .replace('\n', "\\n")
                            .chars()
                            .take(300)
                            .collect::<String>()
                    );
                    yield Ok::<Bytes, std::io::Error>(bytes);
                }};
            }

            let start = json!({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_for_stream,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": input_tokens_start, "output_tokens": 0}
                }
            });
            emit!("message_start", &start);
            emit!("ping", &json!({"type": "ping"}));

            let mut buf: Vec<u8> = Vec::new();
            let mut full_text = String::new();
            let mut stop_reason = "end_turn".to_string();

            'outer: while let Some(chunk) = upstream_stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        println!("[sse-err] upstream stream error: {e}");
                        break 'outer;
                    }
                };
                buf.extend_from_slice(&chunk);

                while let Some(end) = find_event_boundary(&buf) {
                    let event_bytes: Vec<u8> = buf.drain(..end).collect();
                    let event_str = match std::str::from_utf8(&event_bytes) {
                        Ok(s) => s,
                        Err(e) => {
                            println!("[sse-err] non-utf8 frame: {e}");
                            continue;
                        }
                    };

                    for line in event_str.lines() {
                        let line = line.trim_end_matches('\r');
                        if line.is_empty() {
                            continue;
                        }
                        println!("[sse-raw] {}", line);
                        let data = match line.strip_prefix("data:") {
                            Some(d) => d.trim(),
                            None => continue,
                        };
                        if data.is_empty() {
                            continue;
                        }
                        if data == "[DONE]" {
                            println!("[sse-raw] got [DONE]");
                            continue;
                        }
                        let json_val: Value = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(e) => {
                                println!("[sse-err] json parse: {e} | data={}", data.chars().take(200).collect::<String>());
                                continue;
                            }
                        };

                        if let Some(choices) = json_val.get("choices").and_then(|c| c.as_array()) {
                            for choice in choices {
                                if let Some(content) = choice
                                    .get("delta")
                                    .and_then(|d| d.get("content"))
                                    .and_then(|c| c.as_str())
                                {
                                    full_text.push_str(content);
                                }
                                if let Some(fr) =
                                    choice.get("finish_reason").and_then(|f| f.as_str())
                                {
                                    stop_reason = map_stop_reason(fr).to_string();
                                }
                            }
                        }
                    }
                }
            }

            println!(
                "[sse-buffered] {} chars: {}",
                full_text.len(),
                full_text.chars().take(500).collect::<String>()
            );

            let segments = parse_segments(&full_text);
            println!("[sse-segments] {} segments", segments.len());

            let mut has_tool_use = false;
            for (idx, seg) in segments.iter().enumerate() {
                match seg {
                    Segment::Text(text) => {
                        if text.is_empty() {
                            continue;
                        }
                        let cb_start = json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""}
                        });
                        emit!("content_block_start", &cb_start);

                        let delta = json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "text_delta", "text": text}
                        });
                        emit!("content_block_delta", &delta);

                        emit!(
                            "content_block_stop",
                            &json!({"type": "content_block_stop", "index": idx})
                        );
                    }
                    Segment::ToolCall { name, input } => {
                        has_tool_use = true;
                        let uuid = uuid::Uuid::new_v4().simple().to_string();
                        let tool_id = format!("toolu_{}", &uuid[..8]);
                        let cb_start = json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": name,
                                "input": {}
                            }
                        });
                        emit!("content_block_start", &cb_start);

                        let partial = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                        let delta = json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": partial}
                        });
                        emit!("content_block_delta", &delta);

                        emit!(
                            "content_block_stop",
                            &json!({"type": "content_block_stop", "index": idx})
                        );
                    }
                }
            }

            if has_tool_use {
                stop_reason = "tool_use".to_string();
            }

            let output_tokens = tokenize_len(&state_for_stream.tokenizer, &full_text);
            println!("[messages] computed output_tokens: {}", output_tokens);

            let msg_delta = json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"input_tokens": input_tokens_start, "output_tokens": output_tokens}
            });
            emit!("message_delta", &msg_delta);

            emit!("message_stop", &json!({"type": "message_stop"}));

            println!("[sse-done] {} chunks", emit_count);
        };

        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .body(Body::from_stream(translated))
            .unwrap_or_else(|_| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::empty())
                    .unwrap()
            })
    } else {
        println!("[messages] buffering non-streaming response");
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                println!("[messages] failed to read response body: {e}");
                return (StatusCode::BAD_GATEWAY, format!("Upstream read error: {e}"))
                    .into_response();
            }
        };
        println!(
            "[messages] upstream response body ({} bytes): {}",
            bytes.len(),
            String::from_utf8_lossy(&bytes)
                .chars()
                .take(500)
                .collect::<String>()
        );
        let openai_resp: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                return (StatusCode::BAD_GATEWAY, format!("Upstream JSON error: {e}"))
                    .into_response();
            }
        };

        let choice0 = openai_resp
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first());
        let text = choice0
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        let finish_reason = choice0
            .and_then(|c| c.get("finish_reason"))
            .and_then(|f| f.as_str())
            .unwrap_or("stop");
        let id = openai_resp
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| format!("msg_{}", s))
            .unwrap_or_else(|| "msg_unknown".to_string());
        let output_tokens = tokenize_len(&state.tokenizer, &text);

        let anthropic_resp = json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model,
            "stop_reason": map_stop_reason(finish_reason),
            "stop_sequence": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
        });

        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(anthropic_resp.to_string()))
            .unwrap_or_else(|_| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::empty())
                    .unwrap()
            })
    }
}

async fn proxy(State(state): State<Arc<AppState>>, req: Request<Body>) -> impl IntoResponse {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();

    let path = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let url = format!("{}{}", state.upstream, path);

    println!("[proxy] {} {}", &method, path);
    println!("[proxy] request headers:");
    for (k, v) in headers.iter() {
        println!("  {}: {}", k, v.to_str().unwrap_or("<binary>"));
    }

    // validate incoming key if configured
    if let Some(expected) = &state.api_key {
        let provided = headers
            .get("x-api-key")
            .or_else(|| headers.get("authorization"))
            .and_then(|v| v.to_str().ok())
            .map(|v| v.trim_start_matches("Bearer ").trim());

        match provided {
            Some(key) if key == expected.as_str() => {}
            _ => {
                println!("[proxy] unauthorized request to {}", path);
                return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
            }
        }
    }

    // build upstream request, swapping auth header
    let mut upstream_req = state.client.request(method.clone(), &url);
    for (key, value) in headers.iter() {
        if key == "host" || key == "x-api-key" || key == "authorization" || key == "anthropic-beta"
        {
            continue;
        }
        upstream_req = upstream_req.header(key, value);
    }

    let body_bytes = axum::body::to_bytes(req.into_body(), usize::MAX).await;
    match &body_bytes {
        Ok(b) => println!(
            "[proxy] request body ({} bytes): {}",
            b.len(),
            String::from_utf8_lossy(b)
                .chars()
                .take(500)
                .collect::<String>()
        ),
        Err(e) => println!("[proxy] failed to read request body: {e}"),
    }
    let body_bytes = match body_bytes {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("Body read error: {e}")).into_response();
        }
    };

    let mut upstream_req = state.client.request(method, &url);
    for (key, value) in headers.iter() {
        if key == "host" {
            continue;
        }
        upstream_req = upstream_req.header(key, value);
    }
    upstream_req = upstream_req.body(body_bytes);

    // inject vMLX key in the format it expects
    if let Some(upstream_key) = &state.upstream_api_key {
        upstream_req = upstream_req.header("authorization", format!("Bearer {}", upstream_key));
    }

    println!("[proxy] sending to upstream: {}", url);
    match upstream_req.send().await {
        Ok(resp) => {
            let status = resp.status();
            println!("[proxy] upstream status: {}", status);
            println!("[proxy] upstream response headers:");
            for (k, v) in resp.headers().iter() {
                println!("  {}: {}", k, v.to_str().unwrap_or("<binary>"));
            }

            let is_streaming = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(|v| v.contains("text/event-stream"))
                .unwrap_or(false);

            println!("[proxy] is_streaming: {}", is_streaming);

            let status_code =
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let resp_headers = resp.headers().clone();

            if is_streaming {
                println!("[proxy] piping as stream");
                let mut response = Response::builder().status(status_code);
                for (key, value) in resp_headers.iter() {
                    if key == "content-length" || key == "transfer-encoding" {
                        continue;
                    }
                    response = response.header(key, value);
                }
                let stream = resp.bytes_stream();
                response
                    .body(Body::from_stream(stream))
                    .unwrap_or_else(|_| {
                        Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(Body::empty())
                            .unwrap()
                    })
            } else {
                println!("[proxy] buffering non-streaming response");
                match resp.bytes().await {
                    Ok(bytes) => {
                        println!(
                            "[proxy] response body ({} bytes): {}",
                            bytes.len(),
                            String::from_utf8_lossy(&bytes)
                                .chars()
                                .take(500)
                                .collect::<String>()
                        );
                        let mut response = Response::builder().status(status_code);
                        for (key, value) in resp_headers.iter() {
                            if key == "content-length" || key == "transfer-encoding" {
                                continue;
                            }
                            response = response.header(key, value);
                        }
                        response.body(Body::from(bytes)).unwrap_or_else(|_| {
                            Response::builder()
                                .status(StatusCode::INTERNAL_SERVER_ERROR)
                                .body(Body::empty())
                                .unwrap()
                        })
                    }
                    Err(e) => {
                        println!("[proxy] failed to read response body: {e}");
                        (StatusCode::BAD_GATEWAY, format!("Upstream read error: {e}"))
                            .into_response()
                    }
                }
            }
        }
        Err(e) => {
            println!("[proxy] upstream request failed: {e}");
            (StatusCode::BAD_GATEWAY, format!("Upstream error: {e}")).into_response()
        }
    }
}

#[tokio::main]
async fn main() {
    let tokenizer_path =
        env::var("TOKENIZER_PATH").unwrap_or_else(|_| "tokenizer.json".to_string());
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8001);
    let upstream = env::var("UPSTREAM_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());
    let api_key = env::var("API_KEY").ok();
    let upstream_api_key = env::var("UPSTREAM_API_KEY").ok();

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer from {tokenizer_path}: {e}"));

    // No .timeout() call: reqwest treats absence as "no overall deadline",
    // which is what we want for long-lived SSE streams.
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .build()
        .expect("Failed to build reqwest client");

    let state = Arc::new(AppState {
        tokenizer,
        client,
        upstream,
        api_key,
        upstream_api_key,
    });

    let app = Router::new()
        .route("/", get(|| async { StatusCode::OK }))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/messages", post(messages))
        .fallback(proxy)
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("Failed to bind");

    println!(
        "qwen-proxy listening on port {port}, upstream: {}",
        state.upstream
    );
    axum::serve(listener, app).await.expect("Server error");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_stop_reason_known() {
        assert_eq!(map_stop_reason("stop"), "end_turn");
        assert_eq!(map_stop_reason("length"), "max_tokens");
        assert_eq!(map_stop_reason("tool_calls"), "tool_use");
        assert_eq!(map_stop_reason("weird"), "end_turn");
    }

    #[test]
    fn find_event_boundary_lf() {
        let buf = b"data: hi\n\nrest";
        assert_eq!(find_event_boundary(buf), Some(10));
    }

    #[test]
    fn find_event_boundary_crlf() {
        let buf = b"data: hi\r\n\r\nrest";
        assert_eq!(find_event_boundary(buf), Some(12));
    }

    #[test]
    fn find_event_boundary_none() {
        let buf = b"data: incomplete";
        assert_eq!(find_event_boundary(buf), None);
    }

    #[test]
    fn parse_tool_call_single_attr() {
        let (name, input) = parse_tool_call("Read file=\"/tmp/x.txt\"").unwrap();
        assert_eq!(name, "Read");
        assert_eq!(input["file"], json!("/tmp/x.txt"));
    }

    #[test]
    fn parse_tool_call_multi_attr() {
        let (name, input) =
            parse_tool_call("Write file=\"/a.txt\" content=\"hello world\"").unwrap();
        assert_eq!(name, "Write");
        assert_eq!(input["file"], json!("/a.txt"));
        assert_eq!(input["content"], json!("hello world"));
    }

    #[test]
    fn parse_tool_call_escaped_quote() {
        let (_, input) = parse_tool_call("Write content=\"say \\\"hi\\\"\"").unwrap();
        assert_eq!(input["content"], json!("say \"hi\""));
    }

    #[test]
    fn parse_tool_call_escaped_newline() {
        let (_, input) = parse_tool_call("Write content=\"a\\nb\"").unwrap();
        assert_eq!(input["content"], json!("a\nb"));
    }

    #[test]
    fn parse_tool_call_name_with_leading_newline() {
        let (name, input) =
            parse_tool_call("\n  Read file=\"/x\"  \n").unwrap();
        assert_eq!(name, "Read");
        assert_eq!(input["file"], json!("/x"));
    }

    #[test]
    fn parse_segments_text_only() {
        let segs = parse_segments("just some text");
        assert_eq!(segs.len(), 1);
        match &segs[0] {
            Segment::Text(t) => assert_eq!(t, "just some text"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn parse_segments_tool_only() {
        let segs = parse_segments("<tool_call>\nRead file=\"/x\"\n</tool_call>");
        assert_eq!(segs.len(), 1);
        match &segs[0] {
            Segment::ToolCall { name, input } => {
                assert_eq!(name, "Read");
                assert_eq!(input["file"], json!("/x"));
            }
            _ => panic!("expected tool call"),
        }
    }

    #[test]
    fn parse_segments_unterminated_tool_call() {
        let segs = parse_segments("prefix<tool_call>\nRead file=\"/x\"\n");
        assert_eq!(segs.len(), 2);
        match &segs[0] {
            Segment::Text(t) => assert_eq!(t, "prefix"),
            _ => panic!("expected text"),
        }
        match &segs[1] {
            Segment::ToolCall { name, input } => {
                assert_eq!(name, "Read");
                assert_eq!(input["file"], json!("/x"));
            }
            _ => panic!("expected tool_call"),
        }
    }

    #[test]
    fn parse_segments_unterminated_tool_call_only() {
        let segs = parse_segments("<tool_call>Write file=\"/a\" content=\"hi\"");
        assert_eq!(segs.len(), 1);
        match &segs[0] {
            Segment::ToolCall { name, input } => {
                assert_eq!(name, "Write");
                assert_eq!(input["file"], json!("/a"));
                assert_eq!(input["content"], json!("hi"));
            }
            _ => panic!("expected tool_call"),
        }
    }

    #[test]
    fn parse_segments_mixed() {
        let input = "prefix<tool_call>Read file=\"/a\"</tool_call>middle<tool_call>Read file=\"/b\"</tool_call>suffix";
        let segs = parse_segments(input);
        assert_eq!(segs.len(), 5);
        assert!(matches!(&segs[0], Segment::Text(t) if t == "prefix"));
        assert!(matches!(&segs[1], Segment::ToolCall { name, .. } if name == "Read"));
        assert!(matches!(&segs[2], Segment::Text(t) if t == "middle"));
        assert!(matches!(&segs[3], Segment::ToolCall { .. }));
        assert!(matches!(&segs[4], Segment::Text(t) if t == "suffix"));
    }

    #[test]
    fn anthropic_to_openai_flattens_content_blocks() {
        let req = json!({
            "model": "claude-sonnet-4-6",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"}
                ]}
            ],
            "max_tokens": 100
        });
        let out = anthropic_to_openai(&req);
        assert_eq!(out["model"], json!("claude-sonnet-4-6"));
        assert_eq!(out["messages"][0]["role"], json!("system"));
        assert_eq!(out["messages"][0]["content"], json!("You are helpful."));
        assert_eq!(out["messages"][1]["role"], json!("user"));
        assert_eq!(out["messages"][1]["content"], json!("hello \nworld"));
        assert_eq!(out["max_tokens"], json!(100));
    }

    #[test]
    fn anthropic_to_openai_system_blocks() {
        let req = json!({
            "model": "m",
            "system": [{"type": "text", "text": "sysA"}, {"type": "text", "text": "sysB"}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        let out = anthropic_to_openai(&req);
        assert_eq!(out["messages"][0]["content"], json!("sysAsysB"));
    }

    #[test]
    fn anthropic_to_openai_caps_max_tokens() {
        let req = json!({
            "model": "m",
            "messages": [],
            "max_tokens": 1_000_000u64
        });
        let out = anthropic_to_openai(&req);
        assert_eq!(out["max_tokens"], json!(65536));
    }

    #[test]
    fn anthropic_full_text_string_content() {
        let req = json!({
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hello world"}]
        });
        let text = anthropic_full_text(&req);
        assert!(text.contains("You are helpful."));
        assert!(text.contains("hello world"));
    }

    #[test]
    fn anthropic_full_text_text_blocks() {
        let req = json!({
            "system": [{"type": "text", "text": "sysA"}, {"type": "text", "text": "sysB"}],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "piece1"},
                    {"type": "text", "text": "piece2"}
                ]
            }]
        });
        let text = anthropic_full_text(&req);
        assert!(text.contains("sysA"));
        assert!(text.contains("sysB"));
        assert!(text.contains("piece1"));
        assert!(text.contains("piece2"));
    }

    #[test]
    fn anthropic_full_text_tool_result_string() {
        let big = "x".repeat(20_000);
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc", "content": big.clone()}
                ]
            }]
        });
        let text = anthropic_full_text(&req);
        assert!(text.contains(&big));
        assert!(text.len() >= 20_000);
    }

    #[test]
    fn anthropic_full_text_tool_result_blocks() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc", "content": [
                        {"type": "text", "text": "file contents here"}
                    ]}
                ]
            }]
        });
        let text = anthropic_full_text(&req);
        assert!(text.contains("file contents here"));
    }

    #[test]
    fn anthropic_full_text_tool_use_included() {
        let req = json!({
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file": "/x"}}
                ]
            }]
        });
        let text = anthropic_full_text(&req);
        assert!(text.contains("Read"));
        assert!(text.contains("/x"));
    }

    #[test]
    fn anthropic_to_openai_assistant_tool_use() {
        let req = json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me read"},
                    {"type": "tool_use", "id": "toolu_abc", "name": "Read",
                     "input": {"file": "/path"}}
                ]
            }]
        });
        let out = anthropic_to_openai(&req);
        let msg = &out["messages"][0];
        assert_eq!(msg["role"], json!("assistant"));
        assert_eq!(msg["content"], json!("let me read"));
        let tc = &msg["tool_calls"][0];
        assert_eq!(tc["id"], json!("toolu_abc"));
        assert_eq!(tc["type"], json!("function"));
        assert_eq!(tc["function"]["name"], json!("Read"));
        let args: Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"file": "/path"}));
    }

    #[test]
    fn anthropic_to_openai_assistant_tool_use_only() {
        let req = json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file": "/a"}}
                ]
            }]
        });
        let out = anthropic_to_openai(&req);
        let msg = &out["messages"][0];
        assert!(msg["content"].is_null());
        assert_eq!(msg["tool_calls"][0]["function"]["name"], json!("Read"));
    }

    #[test]
    fn anthropic_to_openai_user_tool_result_string() {
        let req = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc",
                     "content": "file contents here"}
                ]
            }]
        });
        let out = anthropic_to_openai(&req);
        let msg = &out["messages"][0];
        assert_eq!(msg["role"], json!("tool"));
        assert_eq!(msg["tool_call_id"], json!("toolu_abc"));
        assert_eq!(msg["content"], json!("file contents here"));
    }

    #[test]
    fn anthropic_to_openai_user_tool_result_blocks() {
        let req = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc", "content": [
                        {"type": "text", "text": "line1"},
                        {"type": "text", "text": "line2"}
                    ]}
                ]
            }]
        });
        let out = anthropic_to_openai(&req);
        let msg = &out["messages"][0];
        assert_eq!(msg["role"], json!("tool"));
        assert_eq!(msg["content"], json!("line1\nline2"));
    }

    #[test]
    fn anthropic_to_openai_user_mixed_text_and_tool_results() {
        let req = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result1"},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "result2"},
                    {"type": "text", "text": "after"}
                ]
            }]
        });
        let out = anthropic_to_openai(&req);
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], json!("user"));
        assert_eq!(msgs[0]["content"], json!("before"));
        assert_eq!(msgs[1]["role"], json!("tool"));
        assert_eq!(msgs[1]["tool_call_id"], json!("toolu_1"));
        assert_eq!(msgs[1]["content"], json!("result1"));
        assert_eq!(msgs[2]["role"], json!("tool"));
        assert_eq!(msgs[2]["tool_call_id"], json!("toolu_2"));
        assert_eq!(msgs[3]["role"], json!("user"));
        assert_eq!(msgs[3]["content"], json!("after"));
    }

    #[test]
    fn anthropic_to_openai_full_tool_roundtrip() {
        let req = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "read /x please"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file": "/x"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "hello from x"}
                ]}
            ]
        });
        let out = anthropic_to_openai(&req);
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], json!("user"));
        assert_eq!(msgs[1]["role"], json!("assistant"));
        assert_eq!(msgs[1]["tool_calls"][0]["id"], json!("toolu_1"));
        assert_eq!(msgs[2]["role"], json!("tool"));
        assert_eq!(msgs[2]["tool_call_id"], json!("toolu_1"));
        assert_eq!(msgs[2]["content"], json!("hello from x"));
    }

    #[test]
    fn anthropic_to_openai_skips_empty_system() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let out = anthropic_to_openai(&req);
        assert_eq!(out["messages"][0]["role"], json!("user"));
    }
}
