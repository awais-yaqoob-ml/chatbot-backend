from datetime import datetime


def save_message(
    client,
    session_id,
    role,
    content,
    agent_used="",
):
    collection = client.collections.get(
        "ChatHistory"
    )

    collection.data.insert(
        properties={
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_used": agent_used,
        }
    )


def get_history(
    client,
    session_id,
    limit=20,
):
    collection = client.collections.get(
        "ChatHistory"
    )

    response = collection.query.fetch_objects(
        limit=1000
    )

    messages = []

    for obj in response.objects:
        props = obj.properties

        if props["session_id"] == session_id:
            messages.append(props)

    messages.sort(
        key=lambda x: x["timestamp"]
    )

    return messages[-limit:]


def delete_history(
    client,
    session_id,
):
    collection = client.collections.get(
        "ChatHistory"
    )

    response = collection.query.fetch_objects(
        limit=10000
    )

    deleted = 0

    for obj in response.objects:
        if (
            obj.properties["session_id"]
            == session_id
        ):
            collection.data.delete_by_id(
                obj.uuid
            )
            deleted += 1

    return deleted


def list_sessions(client):
    collection = client.collections.get(
        "ChatHistory"
    )

    response = collection.query.fetch_objects(
        limit=10000
    )

    sessions = set()

    for obj in response.objects:
        sessions.add(
            obj.properties["session_id"]
        )

    return sorted(list(sessions))