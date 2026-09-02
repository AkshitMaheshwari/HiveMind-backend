"""
Gmail Tools for Universal Multi-Agent System.
Provides live Gmail operations via the Google Gmail REST API:
- Listing recent / unread inbox messages
- Searching emails with Gmail search syntax
- Reading email threads & full message bodies
- Creating drafts in Gmail Drafts
- Sending live emails
"""
import base64
from email.mime.text import MIMEText
import logging
import os
import re
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

GMAIL_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"


def _get_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Build standard authorization headers for Gmail API."""
    tok = token or os.getenv("GMAIL_TOKEN") or os.getenv("GOOGLE_ACCESS_TOKEN")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if tok:
        tok = tok.strip()
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def _extract_header(headers_list: List[Dict[str, str]], name: str) -> str:
    """Helper to find header value by case-insensitive name."""
    for h in headers_list:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _extract_body_from_payload(payload: Dict[str, Any]) -> str:
    """Recursively extract plain text or HTML body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if body_data:
        try:
            decoded = base64.urlsafe_b64decode(body_data.encode("ASCII")).decode("utf-8", errors="replace")
            if "html" in mime_type.lower():
                decoded = re.sub(r"<style.*?</style>", "", decoded, flags=re.DOTALL)
                decoded = re.sub(r"<script.*?</script>", "", decoded, flags=re.DOTALL)
                decoded = re.sub(r"<[^>]+>", " ", decoded)
                decoded = re.sub(r"\s+", " ", decoded).strip()
            return decoded
        except Exception:
            pass

    parts = payload.get("parts", [])
    text_content = []
    for part in parts:
        part_text = _extract_body_from_payload(part)
        if part_text:
            text_content.append(part_text)

    return "\n".join(text_content).strip()


async def gmail_list_messages(
    query: str = "is:unread",
    max_results: int = 10,
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch a list of messages from Gmail matching the search query.
    Returns list of dicts with id, thread_id, subject, from, date, snippet.
    """
    if not token and not os.getenv("GMAIL_TOKEN") and not os.getenv("GOOGLE_ACCESS_TOKEN"):
        return [{
            "error": "Gmail is not connected. Please connect your Gmail account in Settings & Integrations."
        }]

    headers = _get_headers(token)
    url = f"{GMAIL_BASE_URL}/messages"
    params = {
        "q": query,
        "maxResults": min(max_results, 25),
    }

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.get(url, headers=headers, params=params)
            if res.status_code == 401:
                return [{"error": "Gmail authentication failed or token expired. Please reconnect in Settings."}]
            if res.status_code != 200:
                return [{"error": f"Gmail API error ({res.status_code}): {res.text}"}]

            messages_list = res.json().get("messages", [])
            if not messages_list:
                return []

            results = []
            for msg_item in messages_list[:max_results]:
                msg_id = msg_item.get("id")
                msg_detail_res = await client.get(
                    f"{GMAIL_BASE_URL}/messages/{msg_id}?format=metadata&metadataHeaders=Subject&metadataHeaders=From&metadataHeaders=Date",
                    headers=headers
                )
                if msg_detail_res.status_code == 200:
                    data = msg_detail_res.json()
                    headers_list = data.get("payload", {}).get("headers", [])
                    results.append({
                        "id": msg_id,
                        "thread_id": data.get("threadId"),
                        "subject": _extract_header(headers_list, "Subject") or "(No Subject)",
                        "from": _extract_header(headers_list, "From"),
                        "date": _extract_header(headers_list, "Date"),
                        "snippet": data.get("snippet", ""),
                    })
            return results

    except Exception as e:
        logger.error("gmail_list_messages error: %s", e)
        return [{"error": str(e)}]


async def gmail_search_emails(
    query: str,
    max_results: int = 10,
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search Gmail inbox using standard Gmail search queries (e.g. 'from:boss', 'subject:invoice').
    """
    return await gmail_list_messages(query=query, max_results=max_results, token=token)


async def gmail_read_message(
    message_id: str,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch full body and details of a specific Gmail message by ID.
    """
    if not token and not os.getenv("GMAIL_TOKEN") and not os.getenv("GOOGLE_ACCESS_TOKEN"):
        return {"error": "Gmail is not connected. Please connect your Gmail account in Settings & Integrations."}

    headers = _get_headers(token)
    url = f"{GMAIL_BASE_URL}/messages/{message_id}?format=full"

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                payload = data.get("payload", {})
                headers_list = payload.get("headers", [])
                body_text = _extract_body_from_payload(payload)

                return {
                    "id": data.get("id"),
                    "thread_id": data.get("threadId"),
                    "subject": _extract_header(headers_list, "Subject") or "(No Subject)",
                    "from": _extract_header(headers_list, "From"),
                    "to": _extract_header(headers_list, "To"),
                    "date": _extract_header(headers_list, "Date"),
                    "snippet": data.get("snippet", ""),
                    "body": body_text or data.get("snippet", ""),
                }
            elif res.status_code == 401:
                return {"error": "Gmail authentication expired. Please reconnect in Settings."}
            else:
                return {"error": f"Failed to fetch message ({res.status_code}): {res.text}"}

    except Exception as e:
        logger.error("gmail_read_message error: %s", e)
        return {"error": str(e)}


async def gmail_read_thread(
    thread_id: str,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch all messages in a conversation thread.
    """
    if not token and not os.getenv("GMAIL_TOKEN") and not os.getenv("GOOGLE_ACCESS_TOKEN"):
        return {"error": "Gmail is not connected. Please connect your Gmail account in Settings & Integrations."}

    headers = _get_headers(token)
    url = f"{GMAIL_BASE_URL}/threads/{thread_id}?format=full"

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                messages = []
                for msg in data.get("messages", []):
                    payload = msg.get("payload", {})
                    headers_list = payload.get("headers", [])
                    messages.append({
                        "id": msg.get("id"),
                        "from": _extract_header(headers_list, "From"),
                        "to": _extract_header(headers_list, "To"),
                        "date": _extract_header(headers_list, "Date"),
                        "subject": _extract_header(headers_list, "Subject"),
                        "snippet": msg.get("snippet", ""),
                        "body": _extract_body_from_payload(payload),
                    })

                first_subject = messages[0]["subject"] if messages else "Conversation Thread"
                return {
                    "thread_id": thread_id,
                    "subject": first_subject,
                    "message_count": len(messages),
                    "messages": messages,
                }
            else:
                return {"error": f"Failed to fetch thread ({res.status_code}): {res.text}"}

    except Exception as e:
        logger.error("gmail_read_thread error: %s", e)
        return {"error": str(e)}


async def gmail_create_draft(
    to: str,
    subject: str,
    body: str,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an email draft in the user's Gmail Drafts folder.
    """
    if not token and not os.getenv("GMAIL_TOKEN") and not os.getenv("GOOGLE_ACCESS_TOKEN"):
        return {"error": "Gmail is not connected. Please connect your Gmail account in Settings & Integrations."}

    headers = _get_headers(token)
    url = f"{GMAIL_BASE_URL}/drafts"

    try:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw_b64 = base64.urlsafe_b64encode(message.as_bytes()).decode("ASCII")

        payload = {
            "message": {
                "raw": raw_b64
            }
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code in [200, 201]:
                data = res.json()
                return {
                    "status": "success",
                    "draft_id": data.get("id"),
                    "message_id": data.get("message", {}).get("id"),
                    "to": to,
                    "subject": subject,
                }
            else:
                return {"error": f"Failed to create draft ({res.status_code}): {res.text}"}

    except Exception as e:
        logger.error("gmail_create_draft error: %s", e)
        return {"error": str(e)}


async def gmail_send_email(
    to: str,
    subject: str,
    body: str,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send an email directly through the user's Gmail account.
    """
    if not token and not os.getenv("GMAIL_TOKEN") and not os.getenv("GOOGLE_ACCESS_TOKEN"):
        return {"error": "Gmail is not connected. Please connect your Gmail account in Settings & Integrations."}

    headers = _get_headers(token)
    url = f"{GMAIL_BASE_URL}/messages/send"

    try:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw_b64 = base64.urlsafe_b64encode(message.as_bytes()).decode("ASCII")

        payload = {
            "raw": raw_b64
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code in [200, 201]:
                data = res.json()
                return {
                    "status": "sent",
                    "message_id": data.get("id"),
                    "thread_id": data.get("threadId"),
                    "to": to,
                    "subject": subject,
                }
            else:
                return {"error": f"Failed to send email ({res.status_code}): {res.text}"}

    except Exception as e:
        logger.error("gmail_send_email error: %s", e)
        return {"error": str(e)}
