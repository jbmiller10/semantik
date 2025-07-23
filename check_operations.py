import asyncio
from sqlalchemy import select, text
from shared.database import pg_connection_manager

async def check_operations():
    await pg_connection_manager.initialize()
    async with pg_connection_manager.get_session() as db:
        result = await db.execute(text('''
            SELECT 
                c.name as collection_name,
                o.type,
                o.status,
                o.error_message,
                o.created_at
            FROM operations o
            JOIN collections c ON c.id = o.collection_id
            ORDER BY c.name, o.created_at DESC
            LIMIT 50
        '''))
        
        rows = result.fetchall()
        
        print('Recent Operations:')
        print('=' * 80)
        current_collection = None
        
        for row in rows:
            if current_collection != row.collection_name:
                current_collection = row.collection_name
                print(f'\n{row.collection_name}:')
            
            print(f'  {row.created_at} - {row.type} ({row.status})')
            if row.error_message:
                print(f'    ERROR: {row.error_message[:100]}...')

asyncio.run(check_operations())