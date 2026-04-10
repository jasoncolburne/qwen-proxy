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
            let content = match msg.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(blocks)) => blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => String::new(),
            };
            messages_out.push(json!({"role": role, "content": content}));
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

    let url = format!("{}/v1/chat/completions", state.upstream);
    println!(
        "[messages] translating -> {} (model={}, stream={})",
        url, model, stream_requested
    );
    println!(
        "[messages] translated body: {}",
        openai_req.to_string().chars().take(500).collect::<String>()
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
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            });
            emit!("message_start", &start);

            let cb_start = json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            });
            emit!("content_block_start", &cb_start);

            emit!("ping", &json!({"type": "ping"}));

            let mut buf: Vec<u8> = Vec::new();
            let mut stop_reason = "end_turn".to_string();
            let mut output_tokens: u64 = 0;
            let mut input_tokens: u64 = 0;

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
                                if let Some(delta) = choice.get("delta") {
                                    if let Some(content) =
                                        delta.get("content").and_then(|c| c.as_str())
                                    {
                                        if !content.is_empty() {
                                            let evt = json!({
                                                "type": "content_block_delta",
                                                "index": 0,
                                                "delta": {
                                                    "type": "text_delta",
                                                    "text": content
                                                }
                                            });
                                            emit!("content_block_delta", &evt);
                                        }
                                    }
                                }
                                if let Some(fr) =
                                    choice.get("finish_reason").and_then(|f| f.as_str())
                                {
                                    stop_reason = map_stop_reason(fr).to_string();
                                }
                            }
                        }
                        if let Some(usage) = json_val.get("usage") {
                            if let Some(ct) =
                                usage.get("completion_tokens").and_then(|v| v.as_u64())
                            {
                                output_tokens = ct;
                            }
                            if let Some(pt) =
                                usage.get("prompt_tokens").and_then(|v| v.as_u64())
                            {
                                input_tokens = pt;
                            }
                        }
                    }
                }
            }

            emit!(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0})
            );

            let msg_delta = json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
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
        let input_tokens = openai_resp
            .get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = openai_resp
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

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
