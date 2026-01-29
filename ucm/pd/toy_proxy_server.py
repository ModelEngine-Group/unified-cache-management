# Adapted from https://github.com/vllm-project/vllm/blob/main/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

import argparse
import itertools
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm.logger import init_logger

logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize clients based on mode.
    """
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.worker_clients = []  # For PD-mixed workers

    if global_args.pd_disaggregation:
        # === PD disaggregation ===
        for i, (host, port) in enumerate(global_args.prefiller_instances):
            base_url = f"http://{host}:{port}/v1"
            app.state.prefill_clients.append(
                {
                    "client": httpx.AsyncClient(timeout=None, base_url=base_url),
                    "host": host,
                    "port": port,
                    "id": i,
                }
            )

        for i, (host, port) in enumerate(global_args.decoder_instances):
            base_url = f"http://{host}:{port}/v1"
            app.state.decode_clients.append(
                {
                    "client": httpx.AsyncClient(timeout=None, base_url=base_url),
                    "host": host,
                    "port": port,
                    "id": i,
                }
            )

        app.state.prefill_iterator = itertools.cycle(
            range(len(app.state.prefill_clients))
        )
        app.state.decode_iterator = itertools.cycle(
            range(len(app.state.decode_clients))
        )

        print(
            f"[PD Mode] Initialized {len(app.state.prefill_clients)} prefillers "
            f"and {len(app.state.decode_clients)} decoders."
        )

    else:
        # === PD mix ===
        for i, (host, port) in enumerate(global_args.worker_instances):
            base_url = f"http://{host}:{port}/v1"
            app.state.worker_clients.append(
                {
                    "client": httpx.AsyncClient(timeout=None, base_url=base_url),
                    "host": host,
                    "port": port,
                    "id": i,
                }
            )

        app.state.worker_iterator = itertools.cycle(
            range(len(app.state.worker_clients))
        )
        print(
            f"[Mixed Mode] Initialized {len(app.state.worker_clients)} PD-mixed workers."
        )

    yield

    # Close all clients
    for client_list in [
        app.state.prefill_clients,
        app.state.decode_clients,
        app.state.worker_clients,
    ]:
        for client_info in client_list:
            await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument(
        "--pd-disaggregation",
        action="store_true",
        help="Enable PD disaggregation mode (prefill and decode separation)",
    )
    # For PD mix instances
    parser.add_argument(
        "--worker-hosts",
        "--work-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--worker-ports",
        "--work-port",
        type=int,
        nargs="+",
        default=[8100],
    )

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    if len(args.worker_hosts) != len(args.worker_ports):
        raise ValueError("Number of worker hosts must match number of worker ports")

    # Create instance tuples
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    args.worker_instances = list(
        zip(args.worker_hosts, args.worker_ports)
    )  # Mixed workers

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type:  'worker' 、'prefill' 、'decode'

    Returns:
        The next client to use
    """
    if service_type == "worker":
        worker_idx = next(app.state.worker_iterator)
        return app.state.worker_clients[worker_idx]
    elif service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    return response


async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }

        if global_args.pd_disaggregation:
            # === PD disaggregation logic ===

            # Step 1: Send request to prefiller (to trigger computation and cache KV)
            prefill_client_info = get_next_client(request.app, "prefill")
            prefill_req_data = req_data.copy()
            prefill_req_data["stream"] = False
            prefill_req_data["max_tokens"] = 1
            if "stream_options" in prefill_req_data:
                del prefill_req_data["stream_options"]

            response = await prefill_client_info["client"].post(
                api, json=prefill_req_data, headers=headers
            )
            response.raise_for_status()

            # Step 2: Stream full output from decoder
            decode_client_info = get_next_client(request.app, "decode")

            logger.debug(
                "PD-DISAGG: Prefill={}:{}, Decode={}:{}",
                prefill_client_info["host"],
                prefill_client_info["port"],
                decode_client_info["host"],
                decode_client_info["port"],
            )

            async def generate_stream():
                async for chunk in stream_service_response(
                    decode_client_info, api, req_data, request_id
                ):
                    yield chunk

            return StreamingResponse(generate_stream(), media_type="application/json")

        else:
            # === PD mixed mode: Directly forward the entire stream using round-robin ===
            worker_client_info = get_next_client(request.app, "worker")

            logger.debug(
                "PD-MIXED: Forwarding to {}:{}",
                worker_client_info["host"],
                worker_client_info["port"],
            )

            async def generate_stream():
                async with worker_client_info["client"].stream(
                    "POST", api, json=req_data, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error in proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    if global_args.pd_disaggregation:
        return {
            "status": "ok",
            "mode": "pd-disaggregation",
            "prefill_instances": len(app.state.prefill_clients),
            "decode_instances": len(app.state.decode_clients),
        }
    else:
        return {
            "status": "ok",
            "mode": "pd-mixed",
            "worker_instances": len(app.state.worker_clients),
        }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
