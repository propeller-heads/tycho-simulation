use std::sync::Mutex;

use crate::evm::protocol::rfq::{client::RFQClient, state::RFQState};

pub struct RFQStreamBuilder {
    // example: ("bebop", BebopClient)
    providers: Vec<(String, Mutex<Box<dyn RFQClient>>)>,
}

impl RFQStreamBuilder {
    pub fn new() -> Self {
        Self { providers: Vec::new() }
    }

    pub fn add_provider(&mut self, provider: String, source: Mutex<Box<dyn RFQClient>>) {
        self.providers.push((provider, source));
    }

    pub async fn build(self, tx: tokio::sync::mpsc::Sender<Vec<RFQState>>) {
        let sources: Vec<Mutex<Box<dyn RFQClient>>> = self
            .providers
            .into_iter()
            .map(|(_, source)| source)
            .collect();
        stream_rfq_states(sources, tx).await;
    }
}

pub async fn stream_rfq_states(
    sources: Vec<Mutex<Box<dyn RFQClient>>>,
    tx: tokio::sync::mpsc::Sender<Vec<RFQState>>,
) {
    loop {
        let mut states = Vec::new();

        for source in &sources {
            let mut locked_source = source.lock().unwrap();
            match locked_source.next_price_update().await {
                Ok(prices_data) => {
                    for data in prices_data.iter() {
                        let state = RFQState::new(
                            data.base_token().clone(),
                            data.quote_token().clone(),
                            data.clone_box(),
                            locked_source.clone_box(),
                        );
                        states.push(state);
                    }
                }
                Err(err) => {
                    eprintln!("Error updating price: {:?}", err);
                }
            }
        }
    }
}
