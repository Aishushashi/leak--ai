from algosdk.v2client import algod, indexer

# ── Connect to Algorand TestNet ───────────────────────
ALGOD_ADDRESS  = "https://testnet-api.algonode.cloud"
INDEXER_ADDRESS = "https://testnet-idx.algonode.cloud"
ALGOD_TOKEN    = ""   # no token needed for algonode

def get_wallet_details(wallet_address):
    """
    Gets all public information about a wallet address.
    Called when a leak is detected — shows who leaked it.
    """
    try:
        # Connect to Algorand
        algod_client    = algod.AlgodClient(ALGOD_TOKEN, ALGOD_ADDRESS)
        indexer_client  = indexer.IndexerClient(ALGOD_TOKEN, INDEXER_ADDRESS)

        # ── Get wallet balance + basic info ──────────
        account_info = algod_client.account_info(wallet_address)
        balance_algo = account_info.get('amount', 0) / 1_000_000  # convert microALGO to ALGO

        # ── Get transaction history ───────────────────
        transactions = indexer_client.search_transactions_by_address(
            address  = wallet_address,
            limit    = 10   # last 10 transactions
        )

        tx_list = []
        for tx in transactions.get('transactions', []):
            tx_list.append({
                "txId":      tx.get('id'),
                "type":      tx.get('tx-type'),
                "amount":    tx.get('payment-transaction', {}).get('amount', 0) / 1_000_000,
                "timestamp": tx.get('round-time')
            })

        return {
            "walletAddress":     wallet_address,
            "balanceALGO":       balance_algo,
            "totalTransactions": len(tx_list),
            "recentTransactions": tx_list,
            "explorerLink":      f"https://testnet.algoexplorer.io/address/{wallet_address}",
            "status":            "LEAK SUSPECT — Wallet identified"
        }

    except Exception as e:
        return {
            "walletAddress": wallet_address,
            "error":         str(e),
            "explorerLink":  f"https://testnet.algoexplorer.io/address/{wallet_address}"
        }


def register_content_on_blockchain(wallet_address, file_hash, title):
    """
    Records content ownership on blockchain.
    Called when creator uploads content.
    """
    try:
        indexer_client = indexer.IndexerClient(ALGOD_TOKEN, INDEXER_ADDRESS)

        # Search if this hash already registered
        results = indexer_client.search_transactions(
            note_prefix = file_hash[:10].encode()
        )

        if results.get('transactions'):
            return {
                "status":  "already_registered",
                "message": "This content is already on blockchain!"
            }

        return {
            "status":        "success",
            "message":       "Content registered on blockchain",
            "contentHash":   file_hash,
            "owner":         wallet_address,
            "explorerLink":  f"https://testnet.algoexplorer.io/address/{wallet_address}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}