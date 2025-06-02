import os
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        """Initialize Supabase client with credentials"""
        self.supabase_url = os.getenv('SUPABASE_URL', 'https://ixnbfvyeniksbqcfdmdo.supabase.co')
        self.supabase_key = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4bmJmdnllbmlrc2JxY2ZkbWRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE0MDE3NjgsImV4cCI6MjA0Njk3Nzc2OH0.h4JtVbwtKAe38yvtOLYvZIbhmMy6v2QCVg51Q11ubYg')
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise

    def get_dialogs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch Tenglish dialogs from Supabase for training"""
        try:
            response = self.client.table('dialogs').select('*').limit(limit).execute()
            if response.data:
                logger.info(f"Fetched {len(response.data)} Tenglish dialogs for training")
                return response.data
            else:
                logger.warning("No dialogs found in Supabase")
                return []
        except Exception as e:
            logger.error(f"Error fetching dialogs: {str(e)}")
            return []

    def get_images(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch images from Supabase for training"""
        try:
            # Try images table first
            try:
                response = self.client.table('images').select('*').limit(limit).execute()
                if response.data:
                    logger.info(f"Fetched {len(response.data)} images for training")
                    return response.data
            except:
                pass
            
            # Fallback to emotions table with image URLs
            try:
                response = self.client.table('emotions').select('*').not_.is_('image_url', 'null').limit(limit).execute()
                if response.data:
                    images = [{'id': r['id'], 'image_url': r['image_url'], 'created_at': r.get('created_at', '')} for r in response.data]
                    logger.info(f"Fetched {len(images)} images from emotions table for training")
                    return images
            except:
                pass
                
            # Return sample data for training if no tables exist
            logger.warning("No image data found, using sample URLs for training")
            return self._get_sample_images(limit)
            
        except Exception as e:
            logger.error(f"Error fetching images: {str(e)}")
            return self._get_sample_images(limit)

    def _get_sample_images(self, limit: int) -> List[Dict[str, Any]]:
        """Get sample image URLs for training when no data exists"""
        sample_urls = [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
            "https://images.unsplash.com/photo-1494790108755-2616b612b5bc?w=400",
            "https://images.unsplash.com/photo-1552058544-f2b08422138a?w=400",
            "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?w=400",
            "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400",
            "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400",
            "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400",
            "https://images.unsplash.com/photo-1552374196-c4e7ffc6e126?w=400",
            "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400",
            "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400"
        ]
        
        return [
            {
                'id': f'sample_{i+1}',
                'image_url': sample_urls[i % len(sample_urls)],
                'created_at': '2024-01-01T00:00:00Z'
            }
            for i in range(min(limit, len(sample_urls)))
        ]

    # REMOVED ALL SAVING FUNCTIONALITY
    # NO save_emotion_result()
    # NO get_processed_emotions()
    # NO get_emotion_stats()
    # ONLY FETCHING DATA FOR TRAINING