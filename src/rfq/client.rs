use async_trait::async_trait;
use futures::stream::BoxStream;
use std::time::Duration;
use tycho_client::feed::synchronizer::StateSyncMessage;
use tycho_common::{
    models::protocol::GetAmountOutParams, simulation::indicatively_priced::SignedQuote,
};

use crate::rfq::{errors::RFQError, models::TimestampHeader};

#[async_trait]
pub trait RFQClient: Send + Sync {
    /// Returns a stream of updates tagged with the provider name.
    ///
    /// If `throttle_duration` is provided, the stream will sleep for that duration
    /// after each message to throttle the rate of messages.
    fn stream(
        &self,
        throttle_duration: Option<Duration>,
    ) -> BoxStream<'static, Result<(String, StateSyncMessage<TimestampHeader>), RFQError>>;

    // This method is responsible for fetching the binding quote from the RFQ API. Use sender and
    // receiver from GetAmountOutParams to ask for the quote
    async fn request_binding_quote(
        &self,
        params: &GetAmountOutParams,
    ) -> Result<SignedQuote, RFQError>;
}
