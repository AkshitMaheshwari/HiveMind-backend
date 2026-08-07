import asyncio
from db.supabase_client import db_service

async def test():
    if db_service.client:
        resp = db_service.client.table("tasks").select("*").execute()
        tasks = resp.data
        print("Tasks in Supabase DB:", [t["id"] for t in tasks])
        
        if tasks:
            task_id = tasks[0]["id"]
            user_id = tasks[0]["user_id"]
            
            # Let's try to delete using the client directly
            try:
                # 1. Delete events
                db_service.client.table("task_events").delete().eq("task_id", task_id).execute()
                print("Events deleted")
            except Exception as e:
                print("Events delete error:", e)
                
            try:
                # 2. Delete task
                delete_resp = db_service.client.table("tasks").delete().eq("id", task_id).execute()
                print("Task delete response:", delete_resp)
            except Exception as e:
                print("Task delete error:", e)
                
            # Check if it was deleted
            resp2 = db_service.client.table("tasks").select("*").eq("id", task_id).execute()
            print("Is task still there?", len(resp2.data) > 0)
    else:
        print("Not connected to Supabase")

if __name__ == "__main__":
    asyncio.run(test())
