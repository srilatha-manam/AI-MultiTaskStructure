#Inject dilog data into the text emotion analysis system
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

from shared.util.supabase_client import SupabaseClient
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        """Initialize data ingestion with shared Supabase client"""
        try:
            self.supabase_client = SupabaseClient()
            logger.info("Data ingestion initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data ingestion: {str(e)}")
            raise

    def get_dialogs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get dialogs from Supabase"""
        return self.supabase_client.get_dialogs(limit)
