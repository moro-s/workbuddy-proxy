//! Native upstream adapters. Chat Completions keeps its original byte path.
//!
//! The conversion boundary is deliberately independent of HTTP and SQLite;
//! relay integration owns transport, cancellation and the existing stats guard.
pub use crate::upstream_config as config;
pub mod encode;
pub mod output;
pub mod replay;
pub mod request;
pub mod sse;
pub mod streaming;
pub mod usage;

/// Errors contain static descriptions and structural paths, never input values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub code: &'static str,
    pub param: Option<String>,
    pub status: u16,
}

impl Error {
    pub fn input(code: &'static str, path: impl Into<String>) -> Self {
        Self {
            code,
            param: Some(path.into()),
            status: 400,
        }
    }

    pub fn upstream(code: &'static str) -> Self {
        Self {
            code,
            param: None,
            status: 502,
        }
    }

    pub fn envelope(&self) -> serde_json::Value {
        // The distinct code travels in the human-readable message too. Otherwise
        // every stream-level failure looks identical to the user ("invalid or
        // incomplete"), and the one detail that identifies the cause - which
        // event or item was rejected - stays buried in the proxy's internals.
        let message = match self.status {
            _ if self.code == "upstream_http_error" => "Upstream rejected the request".to_string(),
            _ if self.code == "upstream_connection_error" => {
                "Upstream connection failed".to_string()
            }
            _ if self.code == "upstream_timeout" => "Upstream response timed out".to_string(),
            409 => "Conversation state expired; start a new conversation".to_string(),
            400 => "Request cannot be represented by the selected upstream protocol".to_string(),
            _ => format!(
                "Upstream protocol response was invalid or incomplete ({})",
                self.code
            ),
        };
        serde_json::json!({"error":{
            "message": message,
            "type": if self.status == 400 && self.param.is_some() { "invalid_request_error" } else { "upstream_error" },
            "param":self.param,"code":self.code
        }})
    }
}
