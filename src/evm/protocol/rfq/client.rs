use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};

use crate::{
    evm::protocol::rfq::{indicative_price::IndicativePrice, state::BindingQuote},
    models::GetAmountOutParams,
    protocol::errors::SimulationError,
};

#[async_trait]
pub trait RFQClientSource: Send + Sync {
    async fn next_price_update(&mut self)
        -> Result<Vec<Box<dyn IndicativePrice>>, SimulationError>;

    async fn request_binding_quote(
        &self,
        params: &GetAmountOutParams,
    ) -> Result<BindingQuote, SimulationError>;

    fn clone_box(&self) -> Box<dyn RFQClientSource>;
}

#[async_trait]
pub trait IndicativePriceSource: Send + Sync {
    async fn next_price_update(&mut self)
        -> Result<Vec<Box<dyn IndicativePrice>>, SimulationError>;
}

pub struct HttpPricePoller<P: IndicativePrice + for<'de> serde::Deserialize<'de> + 'static> {
    http: reqwest::Client,
    endpoint: String,
    _phantom: std::marker::PhantomData<P>,
}

impl<P: IndicativePrice + for<'de> serde::Deserialize<'de> + 'static> HttpPricePoller<P> {
    pub fn new(endpoint: String) -> Self {
        Self { http: reqwest::Client::new(), endpoint, _phantom: std::marker::PhantomData }
    }
}

#[async_trait]
impl<P> IndicativePriceSource for HttpPricePoller<P>
where
    P: IndicativePrice + for<'de> serde::Deserialize<'de> + Send + Sync + 'static,
{
    async fn next_price_update(
        &mut self,
    ) -> Result<Vec<Box<dyn IndicativePrice>>, SimulationError> {
        let res = self
            .http
            .get(&self.endpoint)
            .send()
            .await?;
        let parsed = res.json::<P>().await?; // this shouldn't be so easy. as of now the IndicativePrice is for 1 token pair. I think we
                                             // will need some new structure to define the whole msg structure for each RFQ
        Ok(vec![Box::new(parsed)])
    }
}

pub struct SyncStream<P>(Pin<Box<dyn Stream<Item = P> + Send + Sync>>);

impl<P> Stream for SyncStream<P> {
    type Item = P;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.0.as_mut().poll_next(cx)
    }
}

pub struct WebSocketPriceStream<P: IndicativePrice + 'static> {
    stream: SyncStream<P>,
}

impl<P: IndicativePrice + 'static> WebSocketPriceStream<P> {
    pub fn new(stream: Pin<Box<dyn Stream<Item = P> + Send + Sync>>) -> Self {
        Self { stream: SyncStream(stream) }
    }
}

#[async_trait]
impl<P> IndicativePriceSource for WebSocketPriceStream<P>
where
    P: IndicativePrice + Send + Sync + 'static,
{
    async fn next_price_update(
        &mut self,
    ) -> Result<Vec<Box<dyn IndicativePrice>>, SimulationError> {
        if let Some(update) = self.stream.next().await {
            Ok(vec![Box::new(update)])
        } else {
            Err(SimulationError::RecoverableError("WebSocket closed".into()))
        }
    }
}
