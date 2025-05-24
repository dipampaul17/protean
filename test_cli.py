#!/usr/bin/env python3
"""Test script for Protean CLI"""

import sys
import os
sys.path.insert(0, '.')

try:
    from protean.cli import main
    print("✅ CLI import successful")
    
    # Test that click decorators are working
    if hasattr(main, 'commands'):
        print(f"✅ CLI has {len(main.commands)} commands")
        for cmd_name in main.commands:
            print(f"   - {cmd_name}")
    else:
        print("❌ CLI commands not found")
        
    print("\n🎯 Protean CLI is ready!")
    print("Run: poetry run python protean/cli.py --help")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}") 