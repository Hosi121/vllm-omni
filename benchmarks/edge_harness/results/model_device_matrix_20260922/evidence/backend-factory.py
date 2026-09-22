# SPDX-License-Identifier: Apache-2.0
"""Factories for non-vLLM stages owned by the existing StageRuntime."""


def create_graph_client(metadata, config, ledger, reservation):
    from .graph import GraphStageClient

    if config.get("backend", {}).get("name") != "external.graph.v1":
        raise ValueError("graph stage requires backend.name=external.graph.v1")
    return GraphStageClient(metadata, config["backend"], ledger, reservation)
