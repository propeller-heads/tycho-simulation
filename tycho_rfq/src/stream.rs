use std::{collections::HashMap, sync::Mutex};

use chrono::Utc;
use tycho_simulation::protocol::{models::Update, state::ProtocolSim};

use crate::{client::RFQClient, state::RFQState};

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

    pub async fn build(self, tx: tokio::sync::mpsc::Sender<Update>) {
        stream_rfq_states(self.providers, tx).await;
    }
}

pub async fn stream_rfq_states(
    sources: Vec<(String, Mutex<Box<dyn RFQClient>>)>,
    tx: tokio::sync::mpsc::Sender<Update>,
) {
    loop {
        let mut states = HashMap::new();

        for (rfq, source) in &sources {
            let mut locked_source = source.lock().unwrap();
            match locked_source.next_price_update().await {
                Ok(prices_data) => {
                    for data in prices_data.iter() {
                        let state = RFQState::new(data.clone_box(), locked_source.clone_box());
                        states.insert(data.id(), Box::new(state) as Box<dyn ProtocolSim>);
                    }
                }
                Err(err) => {
                    eprintln!("Error updating price: {:?}", err);
                }
            }
        }
        let timestamp = Utc::now()
            .naive_utc()
            .and_utc()
            .timestamp() as u64;

        // TODO: how to handle updated, new or removed states?
        // TODO: how to handle states from different RFQ protocols? (different latency most likely)
        tx.send(Update::new(timestamp, states, HashMap::new()))
            .await
            .unwrap();
    }
}
