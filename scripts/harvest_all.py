#!/usr/bin/env python3
"""
Master Data Harvesting Script for Protean Pattern Discovery
Orchestrates postmortem, Stack Overflow, and OSS DAG collection
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv

# Import our harvesting modules
from scripts.ingest.fetch_postmortems import PostmortemHarvester
from scripts.ingest.fetch_stackoverflow import StackOverflowHarvester  
from scripts.ingest.clone_oss_dags import OSSDAGHarvester
from scripts.ingest.fetch_incident_io import IncidentIOHarvester  # ENHANCEMENT 3
from scripts.ingest.fetch_pagerduty import PagerDutyHarvester  # ENHANCEMENT 4


class MasterHarvester:
    """Orchestrate all data harvesting for pattern discovery"""
    
    def __init__(self):
        load_dotenv()
        
        # Validate required environment variables
        self.github_token = os.getenv("GITHUB_TOKEN")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN not found in environment variables")
        
        # Get configuration
        self.data_dir = os.getenv("DATA_DIR", "data/raw")
        self.max_harvest_time = int(os.getenv("MAX_HARVEST_TIME", 7200))  # 2 hours
        self.max_postmortems = int(os.getenv("MAX_POSTMORTEMS", 1000))
        self.max_stackoverflow_posts = int(os.getenv("MAX_STACKOVERFLOW_POSTS", 5000))
        self.max_oss_repos = int(os.getenv("OSS_REPOS_LIMIT", 100))
        
        # Initialize harvesters
        self.postmortem_harvester = PostmortemHarvester(self.github_token, f"{self.data_dir}/postmortems")
        self.stackoverflow_harvester = StackOverflowHarvester(f"{self.data_dir}/stackoverflow")
        self.oss_dag_harvester = OSSDAGHarvester(self.github_token, f"{self.data_dir}/oss_dags")
        self.incident_io_harvester = IncidentIOHarvester(f"{self.data_dir}/incident_io")  # ENHANCEMENT 3
        self.pagerduty_harvester = PagerDutyHarvester(f"{self.data_dir}/pagerduty")  # ENHANCEMENT 4
    
    async def harvest_postmortems(self) -> dict:
        """Harvest infrastructure postmortems"""
        logger.info("🔍 Phase 1: Harvesting Infrastructure Postmortems")
        start_time = time.time()
        
        try:
            postmortems = await self.postmortem_harvester.harvest_all(self.max_postmortems)
            output_file = self.postmortem_harvester.save_postmortems(postmortems)
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "count": len(postmortems),
                "output_file": output_file,
                "elapsed_time": elapsed_time,
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ Postmortem harvest failed: {e}")
            return {
                "success": False,
                "count": 0,
                "output_file": None,
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def harvest_stackoverflow(self) -> dict:
        """Harvest Stack Overflow infrastructure Q&A"""
        logger.info("🔍 Phase 2: Harvesting Stack Overflow Data")
        start_time = time.time()
        
        try:
            posts = await self.stackoverflow_harvester.harvest_all(self.max_stackoverflow_posts)
            output_file = self.stackoverflow_harvester.save_posts(posts)
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "count": len(posts),
                "output_file": output_file,
                "elapsed_time": elapsed_time,
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ Stack Overflow harvest failed: {e}")
            return {
                "success": False,
                "count": 0,
                "output_file": None,
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def harvest_oss_dags(self) -> dict:
        """Harvest open source DAGs and workflows"""
        logger.info("🔍 Phase 3: Harvesting OSS DAGs and Workflows")
        start_time = time.time()
        
        try:
            dags = await self.oss_dag_harvester.harvest_all(self.max_oss_repos)
            output_file = self.oss_dag_harvester.save_dags(dags)
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "count": len(dags),
                "output_file": output_file,
                "elapsed_time": elapsed_time,
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ OSS DAG harvest failed: {e}")
            return {
                "success": False,
                "count": 0,
                "output_file": None,
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }

    async def harvest_incident_io(self) -> dict:
        """ENHANCEMENT 3: Harvest incident.io public RSS feed"""
        logger.info("🔍 Phase 4: Harvesting incident.io RSS Feed")
        start_time = time.time()
        
        try:
            posts = await self.incident_io_harvester.harvest_incident_io_feeds(max_posts=60)
            output_file = self.incident_io_harvester.save_posts(posts)
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "count": len(posts),
                "output_file": output_file,
                "elapsed_time": elapsed_time,
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ incident.io harvest failed: {e}")
            return {
                "success": False,
                "count": 0,
                "output_file": None,
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }

    async def harvest_pagerduty(self) -> dict:
        """ENHANCEMENT 4: Harvest PagerDuty postmortem library"""
        logger.info("🔍 Phase 5: Harvesting PagerDuty Postmortem Library")
        start_time = time.time()
        
        try:
            posts = await self.pagerduty_harvester.harvest_pagerduty_content(max_posts=30)
            output_file = self.pagerduty_harvester.save_posts(posts)
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "count": len(posts),
                "output_file": output_file,
                "elapsed_time": elapsed_time,
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ PagerDuty harvest failed: {e}")
            return {
                "success": False,
                "count": 0,
                "output_file": None,
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def run_complete_harvest(self) -> dict:
        """Run the complete data harvesting pipeline"""
        logger.info("🚀 Starting Complete Infrastructure Data Harvest")
        logger.info("📊 Week 1 Mission: Pattern Genome Data Collection")
        
        total_start_time = time.time()
        results = {
            "start_time": datetime.now().isoformat(),
            "phases": {},
            "summary": {}
        }
        
        # Phase 1: Postmortems
        try:
            postmortem_results = await self.harvest_postmortems()
            results["phases"]["postmortems"] = postmortem_results
            
            if postmortem_results["success"]:
                logger.info(f"✅ Phase 1 Complete: {postmortem_results['count']} postmortems in {postmortem_results['elapsed_time']:.1f}s")
            else:
                logger.warning(f"⚠️  Phase 1 Failed: {postmortem_results['error']}")
        except Exception as e:
            logger.error(f"❌ Phase 1 Critical Error: {e}")
            results["phases"]["postmortems"] = {
                "success": False, "count": 0, "error": str(e), 
                "elapsed_time": 0, "output_file": None
            }
        
        # Phase 2: Stack Overflow (if we still have time)
        elapsed_so_far = time.time() - total_start_time
        if elapsed_so_far < self.max_harvest_time * 0.8:  # Save 20% time for OSS DAGs
            try:
                stackoverflow_results = await self.harvest_stackoverflow()
                results["phases"]["stackoverflow"] = stackoverflow_results
                
                if stackoverflow_results["success"]:
                    logger.info(f"✅ Phase 2 Complete: {stackoverflow_results['count']} posts in {stackoverflow_results['elapsed_time']:.1f}s")
                else:
                    logger.warning(f"⚠️  Phase 2 Failed: {stackoverflow_results['error']}")
            except Exception as e:
                logger.error(f"❌ Phase 2 Critical Error: {e}")
                results["phases"]["stackoverflow"] = {
                    "success": False, "count": 0, "error": str(e),
                    "elapsed_time": 0, "output_file": None
                }
        else:
            logger.warning("⏱️  Skipping Stack Overflow harvest due to time constraints")
            results["phases"]["stackoverflow"] = {
                "success": False, "count": 0, "error": "Skipped due to time constraints",
                "elapsed_time": 0, "output_file": None
            }
        
        # Phase 3: incident.io RSS Feed (if we still have time)
        elapsed_so_far = time.time() - total_start_time
        if elapsed_so_far < self.max_harvest_time * 0.7:  # Use time efficiently
            try:
                incident_io_results = await self.harvest_incident_io()
                results["phases"]["incident_io"] = incident_io_results
                
                if incident_io_results["success"]:
                    logger.info(f"✅ Phase 3 Complete: {incident_io_results['count']} incident.io posts in {incident_io_results['elapsed_time']:.1f}s")
                else:
                    logger.warning(f"⚠️  Phase 3 Failed: {incident_io_results['error']}")
            except Exception as e:
                logger.error(f"❌ Phase 3 Critical Error: {e}")
                results["phases"]["incident_io"] = {
                    "success": False, "count": 0, "error": str(e),
                    "elapsed_time": 0, "output_file": None
                }
        else:
            logger.warning("⏱️  Skipping incident.io harvest due to time constraints")
            results["phases"]["incident_io"] = {
                "success": False, "count": 0, "error": "Skipped due to time constraints",
                "elapsed_time": 0, "output_file": None
            }

        # Phase 4: PagerDuty Postmortem Library (if we still have time)
        elapsed_so_far = time.time() - total_start_time
        if elapsed_so_far < self.max_harvest_time * 0.8:
            try:
                pagerduty_results = await self.harvest_pagerduty()
                results["phases"]["pagerduty"] = pagerduty_results
                
                if pagerduty_results["success"]:
                    logger.info(f"✅ Phase 4 Complete: {pagerduty_results['count']} PagerDuty posts in {pagerduty_results['elapsed_time']:.1f}s")
                else:
                    logger.warning(f"⚠️  Phase 4 Failed: {pagerduty_results['error']}")
            except Exception as e:
                logger.error(f"❌ Phase 4 Critical Error: {e}")
                results["phases"]["pagerduty"] = {
                    "success": False, "count": 0, "error": str(e),
                    "elapsed_time": 0, "output_file": None
                }
        else:
            logger.warning("⏱️  Skipping PagerDuty harvest due to time constraints")
            results["phases"]["pagerduty"] = {
                "success": False, "count": 0, "error": "Skipped due to time constraints",
                "elapsed_time": 0, "output_file": None
            }

        # Phase 5: OSS DAGs (if we still have time)
        elapsed_so_far = time.time() - total_start_time
        if elapsed_so_far < self.max_harvest_time * 0.9:  # Use remaining time
            try:
                oss_dag_results = await self.harvest_oss_dags()
                results["phases"]["oss_dags"] = oss_dag_results
                
                if oss_dag_results["success"]:
                    logger.info(f"✅ Phase 5 Complete: {oss_dag_results['count']} DAGs in {oss_dag_results['elapsed_time']:.1f}s")
                else:
                    logger.warning(f"⚠️  Phase 5 Failed: {oss_dag_results['error']}")
            except Exception as e:
                logger.error(f"❌ Phase 5 Critical Error: {e}")
                results["phases"]["oss_dags"] = {
                    "success": False, "count": 0, "error": str(e),
                    "elapsed_time": 0, "output_file": None
                }
        else:
            logger.warning("⏱️  Skipping OSS DAG harvest due to time constraints")
            results["phases"]["oss_dags"] = {
                "success": False, "count": 0, "error": "Skipped due to time constraints",
                "elapsed_time": 0, "output_file": None
            }
        
        # Calculate summary
        total_elapsed = time.time() - total_start_time
        total_items = sum(phase["count"] for phase in results["phases"].values())
        successful_phases = sum(1 for phase in results["phases"].values() if phase["success"])
        
        results["summary"] = {
            "total_elapsed_time": total_elapsed,
            "total_items_harvested": total_items,
            "successful_phases": successful_phases,
            "total_phases": len(results["phases"]),
            "items_per_second": total_items / total_elapsed if total_elapsed > 0 else 0,
            "end_time": datetime.now().isoformat()
        }
        
        # Log final summary
        logger.info("=" * 60)
        logger.info("🎯 HARVEST COMPLETE - PATTERN GENOME DATA READY")
        logger.info("=" * 60)
        logger.info(f"📊 Total Items: {total_items}")
        logger.info(f"⏱️  Total Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"✅ Success Rate: {successful_phases}/{len(results['phases'])} phases")
        logger.info(f"⚡ Speed: {results['summary']['items_per_second']:.1f} items/second")
        
        for phase_name, phase_result in results["phases"].items():
            status = "✅" if phase_result["success"] else "❌"
            logger.info(f"   {status} {phase_name}: {phase_result['count']} items")
        
        logger.info("📁 Data ready for pattern extraction phase!")
        
        return results
    
    def save_harvest_report(self, results: dict) -> str:
        """Save harvest results to file"""
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(self.data_dir) / f"harvest_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📋 Harvest report saved: {report_file}")
        return str(report_file)


async def main():
    """Main entry point for complete data harvest"""
    try:
        harvester = MasterHarvester()
        results = await harvester.run_complete_harvest()
        report_file = harvester.save_harvest_report(results)
        
        # Exit with appropriate code
        if results["summary"]["successful_phases"] > 0:
            logger.info("🚀 Ready for Pattern Extraction Phase!")
            return 0
        else:
            logger.error("❌ No successful harvest phases - check configuration")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Critical harvest failure: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main()) 