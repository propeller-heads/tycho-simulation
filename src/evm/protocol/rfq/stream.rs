use std::{collections::HashMap, sync::Mutex};

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

    pub async fn build(self, tx: tokio::sync::mpsc::Sender<HashMap<String, RFQState>>) {
        stream_rfq_states(self.providers, tx).await;
    }
}

pub async fn stream_rfq_states(
    sources: Vec<(String, Mutex<Box<dyn RFQClient>>)>,
    tx: tokio::sync::mpsc::Sender<HashMap<String, RFQState>>,
) {
    loop {
        let mut states = HashMap::new();

        for (rfq, source) in &sources {
            let mut locked_source = source.lock().unwrap();
            match locked_source.next_price_update().await {
                Ok(prices_data) => {
                    for data in prices_data.iter() {
                        let state = RFQState::new(data.clone_box(), locked_source.clone_box());
                        let id = format!("{} {} {}", rfq, data.base_token(), data.quote_token());
                        states.insert(id, state);
                    }
                }
                Err(err) => {
                    eprintln!("Error updating price: {:?}", err);
                }
            }
        }
        tx.send(states).await.unwrap();
    }
}
