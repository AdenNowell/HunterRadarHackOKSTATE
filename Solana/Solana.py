#HuntTracker x Solana: Payments + Access Pass + Anchors (Devnet)

#One-file FastAPI microservice for hackathon demo.



Features

#- /pay/session    -> create Solana Pay request with reference key

#- /pay/verify     -> verify on-chain transfer (recipient+amount) by reference

#- /passes/bootstrap -> create a 0-decimals SPL "Access Pass" mint (server is mint authority)

#- /passes/mint_after_payment -> on confirmed payment, mint 1 PASS to payer wallet (ATA)

#- /entitlements/check -> does wallet hold >=1 PASS?

#- /anchor/event   -> write tamper-evident event hash via Memo program

#- (Optional) /pay/split/transaction -> unsigned tx that splits payouts across recipients
from typing import Optional, List, Dict, Any



from fastapi import FastAPI, HTTPException, Query

from pydantic import BaseModel, Field, validator



from solana.rpc.api import Client

from solana.publickey import PublicKey

from solana.keypair import Keypair

from solana.transaction import Transaction, TransactionInstruction, AccountMeta

from solana.rpc.types import TxOpts

from solana.system_program import create_account, CreateAccountParams, transfer, TransferParams

from solana.sysvar import SYSVAR_RENT_PUBKEY

from solana.rpc.commitment import Confirmed



from spl.token.constants import TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID, MINT_LEN

from spl.token.instructions import (

    initialize_mint, InitializeMintParams,

    get_associated_token_address, create_associated_token_account,

    mint_to, MintToParams

)



import base58



# ------------------ Config ------------------

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")

MERCHANT_PUBKEY = os.getenv("SOLANA_MERCHANT_PUBKEY")  # base58 recipient

PAYER_SECRET_B58 = os.getenv("SOLANA_PAYER_SECRET_BASE58")  # base58 (fee payer + mint authority)



if not MERCHANT_PUBKEY:

    raise RuntimeError("Missing SOLANA_MERCHANT_PUBKEY")

if not PAYER_SECRET_B58:

    raise RuntimeError("Missing SOLANA_PAYER_SECRET_BASE58")



client = Client(RPC_URL)

PAYER = Keypair.from_secret_key(base58.b58decode(PAYER_SECRET_B58))

MERCHANT = PublicKey(MERCHANT_PUBKEY)



MEMO_PROGRAM_ID = PublicKey("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")

PASS_INFO_FILE = "./pass_mint.json"   # stores current PASS mint address across runs (in Replit FS)



SESSIONS: Dict[str, Dict[str, Any]] = {}  # reference -> expected {amount_lamports, recipient, ...}



app = FastAPI(title="HuntTracker Solana Service (Devnet)", version="1.1.0")



# ------------------ Models ------------------

class SplitItem(BaseModel):

    recipient: str

    percent: float

    @validator("percent")

    def _p(cls, v): 

        if v <= 0 or v > 100: raise ValueError("percent must be (0,100]")

        return v



class CreateSessionBody(BaseModel):

    amount_sol: float = Field(..., gt=0)

    label: Optional[str] = "HuntTracker"

    message: Optional[str] = "Unlock safety features"

    memo: Optional[str] = None



class VerifyResult(BaseModel):

    status: str

    signature: Optional[str] = None

    slot: Optional[int] = None

    reason: Optional[str] = None

    amount_lamports: Optional[int] = None

    recipient: Optional[str] = None



class AnchorBody(BaseModel):

    event_id: str

    event_hash_hex: str

    note: Optional[str] = None



class MintAfterPayBody(BaseModel):

    reference: str

    payer_pubkey: str  # wallet that paid & should receive the PASS token



class SplitTxBody(BaseModel):

    payer_pubkey: str

    amount_sol: float

    split: List[SplitItem]

    @validator("split")

    def _sum100(cls, v):

        s = sum(i.percent for i in v)

        if abs(s - 100.0) > 1e-6: raise ValueError("split percents must sum to 100")

        return v



# ------------------ Helpers ------------------

def lamports(sol: float) -> int:

    return int(round(sol * 1_000_000_000))



def fmt_amount(sol: float) -> str:

    s = f"{sol:.9f}"

    return s.rstrip("0").rstrip(".")



def make_reference() -> PublicKey:

    return Keypair.generate().public_key



def build_solana_pay_url(recipient: PublicKey, amount_sol: float, reference: PublicKey, label: str, message: str, memo: Optional[str]) -> str:

    params = [f"amount={fmt_amount(amount_sol)}", f"reference={str(reference)}", f"label={label}", f"message={message}"]

    if memo: params.append(f"memo={memo}")

    return f"solana:{str(recipient)}?{'&'.join(params)}"



def find_payment_for_reference(reference: PublicKey, expected_recipient: PublicKey, min_lamports: int) -> Optional[VerifyResult]:

    sigs = client.get_signatures_for_address(reference, limit=20).get("result") or []

    for e in sigs:

        sig = e["signature"]

        txr = client.get_transaction(sig, max_supported_transaction_version=0).get("result")

        if not txr: continue

        try:

            msg = txr["transaction"]["message"]

            slot = txr.get("slot")

            keys = [ak["pubkey"] if isinstance(ak, dict) else ak for ak in msg["accountKeys"]]

            if str(reference) not in keys: 

                continue

            total_to_recipient = 0

            for ix in msg.get("instructions", []):

                if ix.get("program") == "system" and ix.get("parsed", {}).get("type") == "transfer":

                    info = ix["parsed"]["info"]

                    if info.get("destination") == str(expected_recipient):

                        total_to_recipient += int(info.get("lamports", 0))

            if total_to_recipient >= min_lamports:

                return VerifyResult(status="CONFIRMED", signature=sig, slot=slot, amount_lamports=total_to_recipient, recipient=str(expected_recipient))

        except Exception:

            continue

    return None



def send_memo(payer: Keypair, memo_text: str) -> str:

    ix = TransactionInstruction(

        program_id=MEMO_PROGRAM_ID,

        data=memo_text.encode("utf-8"),

        keys=[AccountMeta(pubkey=payer.public_key, is_signer=True, is_writable=False)],

    )

    tx = Transaction().add(ix)

    resp = client.send_transaction(tx, payer, opts=TxOpts(skip_preflight=True))

    sig = resp.get("result", {}).get("signature") or resp.get("result")

    if not sig:

        raise RuntimeError(f"Memo send failed: {resp}")

    return sig



def load_pass_info() -> Optional[Dict[str, Any]]:

    if os.path.exists(PASS_INFO_FILE):

        with open(PASS_INFO_FILE, "r") as f:

            return json.load(f)

    return None



def save_pass_info(info: Dict[str, Any]):

    with open(PASS_INFO_FILE, "w") as f:

        json.dump(info, f, indent=2)



def ensure_blockhash(tx: Transaction):

    tx.recent_blockhash = client.get_latest_blockhash()["result"]["value"]["blockhash"]



# ------------------ Endpoints ------------------

@app.get("/health")

def health():

    pi = load_pass_info()

    return {

        "ok": True,

        "network": "devnet",

        "rpc": RPC_URL,

        "merchant": str(MERCHANT),

        "fee_payer": str(PAYER.public_key),

        "pass_mint": (pi or {}).get("mint_pubkey"),

    }



@app.post("/pay/session")

def pay_session(body: CreateSessionBody):

    ref = make_reference()

    lam = lamports(body.amount_sol)

    SESSIONS[str(ref)] = {

        "amount_lamports": lam,

        "amount_sol": body.amount_sol,

        "recipient": str(MERCHANT),

        "label": body.label,

        "message": body.message,

        "memo": body.memo,

        "created_at": int(time.time()),

    }

    url = build_solana_pay_url(MERCHANT, body.amount_sol, ref, body.label or "HuntTracker", body.message or "Unlock", body.memo)

    return {

        "reference": str(ref),

        "recipient": str(MERCHANT),

        "amount_sol": body.amount_sol,

        "solana_pay_url": url,

        "explorer_reference": f"https://explorer.solana.com/address/{ref}?cluster=devnet",

        "note": "Open in a Devnet wallet (Phantom set to Devnet). Then call /pay/verify."

    }



@app.get("/pay/verify", response_model=VerifyResult)

def pay_verify(reference: str = Query(...)):

    session = SESSIONS.get(reference)

    if not session:

        raise HTTPException(404, "Unknown reference")

    found = find_payment_for_reference(PublicKey(reference), PublicKey(session["recipient"]), session["amount_lamports"])

    return found or VerifyResult(status="PENDING", reason="No matching transfer yet.")



@app.post("/anchor/event")

def anchor_event(body: AnchorBody):

    memo_text = f"hunt:event:{body.event_id}:{body.event_hash_hex}"

    if body.note: memo_text += f":{body.note[:64]}"

    sig = send_memo(PAYER, memo_text)

    return {

        "status": "SUBMITTED",

        "signature": sig,

        "explorer": f"https://explorer.solana.com/tx/{sig}?cluster=devnet",

        "memo": memo_text

    }



# ----- PASS (SPL token, decimals=0) -----

@app.post("/passes/bootstrap")

def passes_bootstrap():

    if load_pass_info():

        return {"status": "EXISTS", **load_pass_info()}

    # Create mint account

    mint_kp = Keypair()

    rent = client.get_minimum_balance_for_rent_exemption(MINT_LEN)["result"]

    create_ix = create_account(

        CreateAccountParams(

            from_pubkey=PAYER.public_key,

            new_account_pubkey=mint_kp.public_key,

            lamports=rent,

            space=MINT_LEN,

            program_id=TOKEN_PROGRAM_ID,

        )

    )

    init_ix = initialize_mint(

        InitializeMintParams(

            program_id=TOKEN_PROGRAM_ID,

            mint=mint_kp.public_key,

            decimals=0,  # NFT-like pass

            mint_authority=PAYER.public_key,

            freeze_authority=PAYER.public_key,

        )

    )

    tx = Transaction().add(create_ix, init_ix)

    ensure_blockhash(tx)

    resp = client.send_transaction(tx, PAYER, mint_kp, opts=TxOpts(skip_preflight=True))

    sig = resp.get("result", {}).get("signature") or resp.get("result")

    info = {"mint_pubkey": str(mint_kp.public_key), "tx_signature": sig}

    save_pass_info(info)

    return {"status": "CREATED", **info, "explorer": f"https://explorer.solana.com/address/{mint_kp.public_key}?cluster=devnet"}



@app.post("/passes/mint_after_payment")

def mint_after_payment(body: MintAfterPayBody):

    # 1) Verify payment

    vr = pay_verify(reference=body.reference)

    if vr.status != "CONFIRMED":

        raise HTTPException(400, f"Payment not confirmed: {vr.status}")

    # 2) Load PASS mint

    info = load_pass_info()

    if not info:

        raise HTTPException(400, "PASS mint not initialized. Call /passes/bootstrap first.")

    mint_pk = PublicKey(info["mint_pubkey"])

    owner = PublicKey(body.payer_pubkey)

    ata = get_associated_token_address(owner, mint_pk)

    # 3) Create ATA if missing + Mint 1

    tx = Transaction()

    # Always include ATA create (idempotent—wallets accept creating even if exists)

    tx.add(create_associated_token_account(payer=PAYER.public_key, owner=owner, mint=mint_pk))

    tx.add(mint_to(MintToParams(program_id=TOKEN_PROGRAM_ID, mint=mint_pk, dest=ata, authority=PAYER.public_key, amount=1)))

    ensure_blockhash(tx)

    resp = client.send_transaction(tx, PAYER, opts=TxOpts(skip_preflight=True))

    sig = resp.get("result", {}).get("signature") or resp.get("result")

    return {

        "status": "MINTED",

        "pass_mint": str(mint_pk),

        "recipient": str(owner),

        "ata": str(ata),

        "tx": sig,

        "explorer": f"https://explorer.solana.com/tx/{sig}?cluster=devnet"

    }



@app.get("/entitlements/check")

def entitlements_check(owner: str):

    info = load_pass_info()

    if not info:

        raise HTTPException(400, "PASS mint not initialized.")

    mint_pk = PublicKey(info["mint_pubkey"])

    ata = str(get_associated_token_address(PublicKey(owner), mint_pk))

    bal = client.get_token_account_balance(ata)

    ui = ((bal.get("result") or {}).get("value") or {}).get("uiAmount")

    return {"owner": owner, "pass_mint": str(mint_pk), "ata": ata, "balance": ui or 0}



# ----- Optional: build split payout unsigned tx -----

@app.post("/pay/split/transaction")

def split_tx(body: SplitTxBody):

    payer = PublicKey(body.payer_pubkey)

    total_lam = lamports(body.amount_sol)

    tx = Transaction()

    # Compute lamports per recipient (rounding to first)

    alloc = []

    sofar = 0

    for i, s in enumerate(body.split):

        amt = int(total_lam * (s.percent / 100.0))

        alloc.append((PublicKey(s.recipient), amt))

        sofar += amt

    if alloc:

        r0, a0 = alloc[0]

        alloc[0] = (r0, a0 + (total_lam - sofar))

    for r, amt in alloc:

        tx.add(transfer(TransferParams(from_pubkey=payer, to_pubkey=r, lamports=amt)))

    # Add a memo with a random reference for later verification

    ref = make_reference()

    tx.add(TransactionInstruction(program_id=MEMO_PROGRAM_ID, data=f"split_ref:{str(ref)}".encode(), keys=[AccountMeta(pubkey=payer, is_signer=True, is_writable=False)]))

    tx.fee_payer = payer

    ensure_blockhash(tx)

    b64 = base64.b64encode(tx.serialize(verify_signatures=False)).decode()

    return {

        "unsigned_tx_base64": b64,

        "reference": str(ref),

        "explainer": "Have the wallet sign & send this. Then verify with /pay/verify using the reference."

    }