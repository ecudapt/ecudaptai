#!/usr/bin/env python3
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(dotenv_path='../../.env')

supabase_url = os.getenv("VITE_SUPABASE_URL")
supabase_key = os.getenv("VITE_SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    print("❌ Error: Missing Supabase credentials in .env file")
    exit(1)

supabase = create_client(supabase_url, supabase_key)

print("\n" + "="*60)
print("  ECUdapt AI - Database Test Query")
print("="*60)

try:
    runs = supabase.table("scrape_runs").select("*").order("started_at", desc=True).limit(5).execute()
    print(f"\n📊 Recent Scrape Runs ({len(runs.data)}):")
    if runs.data:
        for run in runs.data:
            status_icon = "✅" if run['status'] == 'completed' else "⏳" if run['status'] == 'running' else "❌"
            print(f"  {status_icon} {run['started_at'][:19]}: {run['total_threads']} threads, {run['total_posts']} posts")
    else:
        print("  No scrape runs found yet")

    threads = supabase.table("forum_threads").select("*").limit(10).execute()
    print(f"\n🧵 Sample Threads ({len(threads.data)}):")
    if threads.data:
        for thread in threads.data:
            title = thread['title'][:60] + "..." if len(thread['title']) > 60 else thread['title']
            print(f"  - [{thread['domain']}] {title}")
    else:
        print("  No threads found yet")

    posts = supabase.table("forum_posts").select("*").limit(5).execute()
    print(f"\n💬 Sample Posts ({len(posts.data)}):")
    if posts.data:
        for post in posts.data:
            content = post['content'][:80] + "..." if len(post['content']) > 80 else post['content']
            print(f"  - {post['author']}: {content}")
    else:
        print("  No posts found yet")

    total_threads = supabase.table("forum_threads").select("*", count="exact").execute()
    total_posts = supabase.table("forum_posts").select("*", count="exact").execute()

    print(f"\n📈 Total Statistics:")
    print(f"  - Total Threads: {total_threads.count}")
    print(f"  - Total Posts: {total_posts.count}")

    print("\n" + "="*60)
    print("✅ Database connection successful!")
    print("="*60 + "\n")

except Exception as e:
    print(f"\n❌ Error querying database: {e}\n")
    exit(1)
