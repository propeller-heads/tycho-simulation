import asyncio
from decimal import Decimal
from logging import getLogger
from typing import Optional

from tycho_indexer_client import TychoStream, TychoRPCClient
import tycho_indexer_client.dto as tycho_models
import os

from tycho_simulation_py.evm.decoders import ThirdPartyPoolTychoDecoder
from tycho_simulation_py.evm.pool_state import ThirdPartyPool
from tycho_simulation_py.models import EthereumToken, EVMBlock

log = getLogger(__name__)


class TokenFactory:
    def __init__(self, tokens: Optional[dict[str, EthereumToken]] = None):
        self.tokens = tokens

    def get_token(self, address: str) -> Optional[EthereumToken]:
        if address not in self.tokens:
            return None
        return self.tokens[address]

    def get_tokens(self, addresses: list[str]) -> list[EthereumToken]:
        return [self.get_token(addr) for addr in addresses]


def load_all_tokens(rpc_client: TychoRPCClient) -> dict[str, EthereumToken]:
    page = 0
    tokens = dict()
    log.info("Loading all tokens from Tycho.")
    while True:
        params = tycho_models.TokensParams(
            pagination=tycho_models.PaginationParams(page=page, page_size=1000),
            min_quality=50,
            traded_n_days_ago=30,
        )

        res = rpc_client.get_tokens(params)
        if not res:
            break

        for token in res:
            address = token.address.hex().lower()
            tokens[address] = EthereumToken(token.symbol, address, token.decimals)

        page += 1
    return tokens


async def run_example():
    curdir = os.path.dirname(os.path.abspath(__file__)) + "/"
    auth_token = os.getenv("TYCHO_AUTH_KEY", "sampletoken")

    tycho_stream = TychoStream(
        "tycho-beta.propellerheads.xyz",
        ["vm:balancer"],
        blockchain=tycho_models.Chain.ethereum,
        auth_token=auth_token,
        include_state=True,
        logs_directory=curdir + "logs",
        min_tvl=Decimal("1000"),
    )
    rpc_client = TychoRPCClient(
        rpc_url="https://tycho-beta.propellerheads.xyz", auth_token=auth_token
    )

    all_tokens = load_all_tokens(rpc_client)

    token_factory = TokenFactory(all_tokens)

    decoder = ThirdPartyPoolTychoDecoder(
        token_factory.get_tokens,
        adapter_contract=curdir + "assets/BalancerV2SwapAdapter.evm.runtime",
        minimum_gas=72300,
        trace=False,
    )

    # Starts Tycho Stream. This will run a subprocess that connects to the Tycho Indexer and streams data.
    await tycho_stream.start()

    pools: dict[str, ThirdPartyPool] = dict()

    removed_pools = set()
    decoded_count = 0
    total_count = 0
    n_new_tokens = 0

    async for msg in tycho_stream:
        for exchange, sync_msg in msg.sync_states.items():
            if sync_msg.status.lower() != "ready":
                log.warning(
                    f"Exchange {exchange} is not ready! Current status is: {sync_msg.status}"
                )
        block: EVMBlock = EVMBlock(
            msg.sync_states["vm:balancer"].header.number,
            msg.sync_states["vm:balancer"].header.hash.hex(),
        )

        for exchange, state_msg in msg.state_msgs.items():
            new_pools = decoder.decode_snapshot(state_msg.snapshots, block)
            updates = decoder.apply_deltas(pools, state_msg.deltas, block)

            log.info(f"Found {len(new_pools)} new pools for {exchange}.")
            log.info(f"Updated {len(updates)} pools for {exchange}.")

            pools.update(new_pools)
            pools.update(updates)
            for pool_id in removed_pools:
                pools.pop(pool_id, None)

            for pool_id, pool in pools.items():
                print(f"Block: {block.id}")
                print(
                    f"Pool {pool_id}. "
                    f"Token0: {pool.tokens[0].symbol} ({pool.tokens[0].address}). "
                    f"Token1: {pool.tokens[1].symbol} ({pool.tokens[1].address}). "
                    f"Spot prices: {pool.marginal_prices}. "
                )


if __name__ == "__main__":
    asyncio.run(run_example())
