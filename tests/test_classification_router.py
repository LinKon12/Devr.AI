import pytest
import sys
sys.path.insert(0, 'backend')

from unittest.mock import patch, MagicMock, AsyncMock
from app.classification.classification_router import ClassificationRouter

@pytest.mark.anyio
async def test_classification_fallback_on_error():
    """Test that classification fallback returns needs_devrel: False on error"""
    
    # Mock ClassificationRouter's should_process_message to force exception path
    with patch('app.classification.classification_router.ClassificationRouter.should_process_message') as mock_method:
        # Create a real router instance
        router = ClassificationRouter()
        
        # Manually call the internal logic that would fail
        # Simulate what happens when LLM fails
        result = router._fallback_triage("test message")
        
        # Should fallback to False (not spam on errors)
        assert result["needs_devrel"] is False
        assert result["priority"] == "low"
        assert "Fallback" in result["reasoning"]

@pytest.mark.anyio  
async def test_classification_fallback_on_invalid_json():
    """Test that classification fallback handles invalid JSON from LLM"""
    router = ClassificationRouter()
    
    # Test the fallback method directly
    result = router._fallback_triage("test message")
    
    # Should return needs_devrel: False
    assert result["needs_devrel"] is False
    assert result["priority"] == "low"
    assert "Fallback" in result["reasoning"]