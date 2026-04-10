use std::env;
use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

struct AppState {
    tokenizer: Tokenizer,
    client: reqwest::Client,
    upstream: String,
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
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Tokenizer error: {e}")).into_response(),
    }
}

async fn proxy(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> impl IntoResponse {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();

    let url = format!(
        "{}{}",
        state.upstream,
        uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/")
    );

    let body_stream = req.into_body().into_data_stream();
    let upstream_body = reqwest::Body::wrap_stream(body_stream);

    let mut upstream_req = state.client.request(method, &url);
    for (key, value) in headers.iter() {
        if key == "host" {
            continue;
        }
        upstream_req = upstream_req.header(key, value);
    }
    upstream_req = upstream_req.body(upstream_body);

    match upstream_req.send().await {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let resp_headers = resp.headers().clone();

            let mut response = Response::builder().status(status);
            for (key, value) in resp_headers.iter() {
                // content-length is invalid once we stream; let the framework handle framing
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
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("Upstream error: {e}")).into_response(),
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
    let upstream =
        env::var("UPSTREAM_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());

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
    });

    let app = Router::new()
        .route("/v1/messages/count_tokens", post(count_tokens))
        .fallback(proxy)
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .expect("Failed to bind");

    println!("qwen-proxy listening on port {port}, upstream: {}", state.upstream);
    axum::serve(listener, app).await.expect("Server error");
}
