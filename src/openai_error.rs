use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
    error_type: String,
    param: Option<String>,
    code: Option<String>,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error".to_string(),
            param: None,
            code: None,
        }
    }

    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            message: message.into(),
            error_type: "not_implemented".to_string(),
            param: None,
            code: None,
        }
    }

    pub fn server_error(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            error_type: "server_error".to_string(),
            param: None,
            code: None,
        }
    }

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let payload = OpenAiErrorResponse {
            error: OpenAiError {
                message: self.message,
                error_type: self.error_type,
                param: self.param,
                code: self.code,
            },
        };
        (self.status, Json(payload)).into_response()
    }
}

#[derive(Serialize)]
struct OpenAiErrorResponse {
    error: OpenAiError,
}

#[derive(Serialize)]
struct OpenAiError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    param: Option<String>,
    code: Option<String>,
}
