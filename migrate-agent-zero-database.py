#!/usr/bin/env python3
"""
🔧 Agent Zero V2.0 - Migration & Database Enhancement Script
🗄️  Rozszerzenie bazy danych o brakujące funkcjonalności Phase 4-9
📅 12 października 2025 | Production Database Migration

ZESPÓŁ: Developer A + AI Assistant  
STATUS: PRODUCTION READY - Complete Database Schema
KOMPATYBILNOŚĆ: Agent Zero V1/V2.0 (zachowuje istniejące tabele)
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

class AgentZeroDatabaseMigrator:
    """Production Database Migration for Agent Zero V2.0 Enhanced Features"""
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self.backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def run_migration(self) -> Dict[str, Any]:
        """Execute Complete Database Migration"""
        print("🚀 Starting Agent Zero V2.0 Database Migration...")
        
        # 1. Create backup
        if os.path.exists(self.db_path):
            print(f"💾 Creating backup: {self.backup_path}")
            os.system(f"cp {self.db_path} {self.backup_path}")
        
        # 2. Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
        
        migration_results = {}
        
        try:
            # 3. Run all migrations
            migration_results["v1_compatibility"] = self._ensure_v1_compatibility(conn)
            migration_results["team_formation"] = self._migrate_team_formation(conn)
            migration_results["analytics"] = self._migrate_analytics(conn)
            migration_results["collaboration"] = self._migrate_collaboration(conn)
            migration_results["predictive"] = self._migrate_predictive(conn)
            migration_results["learning"] = self._migrate_adaptive_learning(conn)
            migration_results["quantum"] = self._migrate_quantum_intelligence(conn)
            migration_results["indexes"] = self._create_performance_indexes(conn)
            migration_results["sample_data"] = self._insert_sample_data(conn)
            
            # 4. Verify migration
            migration_results["verification"] = self._verify_migration(conn)
            
            conn.commit()
            print("✅ Migration completed successfully!")
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Migration failed: {e}")
            migration_results["error"] = str(e)
            raise
        
        finally:
            conn.close()
        
        return migration_results
    
    def _ensure_v1_compatibility(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Ensure V1 tables exist for backward compatibility"""
        print("🔄 Ensuring V1 compatibility...")
        
        v1_tables = """
        -- Original Agent Zero V1 tables (backward compatibility)
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 1,
            agent_id TEXT,
            project_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP NULL
        );
        
        CREATE TABLE IF NOT EXISTS agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            skills TEXT, -- JSON string
            status TEXT DEFAULT 'available',
            cost_per_hour REAL DEFAULT 100.0,
            timezone TEXT DEFAULT 'UTC',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'planning',
            budget REAL,
            deadline DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        conn.executescript(v1_tables)
        return {"status": "v1_tables_ensured", "tables": ["tasks", "agents", "projects"]}
    
    def _migrate_team_formation(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Phase 4: Team Formation & Learning Tables"""
        print("👥 Migrating Team Formation & Learning...")
        
        team_formation_schema = """
        -- Team Formation & Learning (Phase 4)
        CREATE TABLE IF NOT EXISTS team_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            team_composition TEXT NOT NULL, -- JSON: [agent_ids]
            formation_strategy TEXT DEFAULT 'ai_recommended',
            outcome_success REAL NOT NULL CHECK (outcome_success >= 0.0 AND outcome_success <= 1.0),
            budget_delta REAL DEFAULT 0.0, -- Actual vs planned budget %
            timeline_delta REAL DEFAULT 0.0, -- Actual vs planned timeline %
            quality_score REAL DEFAULT 0.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
            team_satisfaction REAL DEFAULT 0.0 CHECK (team_satisfaction >= 0.0 AND team_satisfaction <= 1.0),
            collaboration_rating REAL DEFAULT 0.0,
            lessons_learned TEXT, -- JSON: insights from this team
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project_id)
        );
        
        CREATE TABLE IF NOT EXISTS agent_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            role TEXT NOT NULL,
            individual_score REAL NOT NULL CHECK (individual_score >= 0.0 AND individual_score <= 1.0),
            collaboration_score REAL NOT NULL CHECK (collaboration_score >= 0.0 AND collaboration_score <= 1.0),
            skill_growth REAL DEFAULT 0.0, -- Skill improvement during project
            task_completion_rate REAL DEFAULT 1.0,
            quality_rating REAL DEFAULT 0.0,
            feedback_text TEXT,
            peer_ratings TEXT, -- JSON: {agent_id: rating}
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        );
        
        CREATE TABLE IF NOT EXISTS team_synergy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_a TEXT NOT NULL,
            agent_b TEXT NOT NULL,
            synergy_score REAL NOT NULL CHECK (synergy_score >= 0.0 AND synergy_score <= 1.0),
            project_count INTEGER DEFAULT 1,
            avg_performance REAL NOT NULL,
            communication_score REAL DEFAULT 0.0,
            conflict_incidents INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(agent_a, agent_b),
            CHECK (agent_a != agent_b)
        );
        
        CREATE TABLE IF NOT EXISTS team_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            recommended_team TEXT NOT NULL, -- JSON: [agent_ids]
            recommendation_confidence REAL NOT NULL,
            reasoning TEXT, -- JSON: explanation
            alternative_teams TEXT, -- JSON: backup options
            constraints_considered TEXT, -- JSON: budget, timeline, etc.
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accepted BOOLEAN DEFAULT NULL, -- NULL=pending, TRUE=accepted, FALSE=rejected
            actual_outcome REAL DEFAULT NULL -- If team was used, record outcome
        );
        """
        
        conn.executescript(team_formation_schema)
        return {"status": "team_formation_migrated", "tables": ["team_history", "agent_performance", "team_synergy", "team_recommendations"]}
    
    def _migrate_analytics(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Phase 5: Advanced Analytics & Reporting"""
        print("📊 Migrating Advanced Analytics...")
        
        analytics_schema = """
        -- Advanced Analytics & Reporting (Phase 5)
        CREATE TABLE IF NOT EXISTS analytics_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL, -- 'hubspot', 'powerbi', 'internal'
            category TEXT NOT NULL, -- 'deals', 'performance', 'metrics'
            subcategory TEXT DEFAULT NULL, -- Additional classification
            data_json TEXT NOT NULL, -- JSON blob of the actual data
            metadata_json TEXT, -- Schema, sync info, etc.
            record_count INTEGER DEFAULT 0, -- Number of records in this dataset
            data_quality_score REAL DEFAULT 1.0, -- 0.0-1.0
            sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sync_status TEXT DEFAULT 'success', -- success, partial, failed
            error_log TEXT DEFAULT NULL
        );
        
        CREATE TABLE IF NOT EXISTS analytics_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            report_type TEXT NOT NULL, -- executive, detailed, technical
            report_data_json TEXT NOT NULL, -- Complete report data
            format TEXT NOT NULL, -- xlsx, docx, pdf
            template_used TEXT DEFAULT 'default',
            file_path TEXT,
            file_size_bytes INTEGER DEFAULT 0,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            download_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP DEFAULT NULL
        );
        
        CREATE TABLE IF NOT EXISTS data_connectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            connector_name TEXT UNIQUE NOT NULL, -- hubspot, powerbi, salesforce
            connector_type TEXT NOT NULL, -- crm, bi, database
            config_json TEXT NOT NULL, -- Connection configuration
            status TEXT DEFAULT 'active', -- active, inactive, error
            last_sync TIMESTAMP DEFAULT NULL,
            sync_frequency_hours INTEGER DEFAULT 24, -- How often to sync
            success_count INTEGER DEFAULT 0,
            error_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS kpi_definitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kpi_name TEXT UNIQUE NOT NULL,
            kpi_description TEXT,
            calculation_formula TEXT, -- SQL or JSON formula
            target_value REAL DEFAULT NULL,
            unit TEXT DEFAULT NULL, -- %, $, count, etc.
            category TEXT NOT NULL, -- performance, financial, quality
            update_frequency TEXT DEFAULT 'daily', -- hourly, daily, weekly
            enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        conn.executescript(analytics_schema)
        return {"status": "analytics_migrated", "tables": ["analytics_dataset", "analytics_reports", "data_connectors", "kpi_definitions"]}
    
    def _migrate_collaboration(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Phase 6: Real-Time Collaboration Intelligence"""
        print("🤝 Migrating Real-Time Collaboration...")
        
        collaboration_schema = """
        -- Real-Time Collaboration Intelligence (Phase 6)
        CREATE TABLE IF NOT EXISTS communication_channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT UNIQUE NOT NULL,
            platform TEXT NOT NULL, -- slack, teams, internal
            channel_name TEXT NOT NULL,
            channel_type TEXT DEFAULT 'group', -- dm, group, public
            project_id TEXT DEFAULT NULL,
            team_members TEXT, -- JSON: [agent_ids]
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            active BOOLEAN DEFAULT TRUE
        );
        
        CREATE TABLE IF NOT EXISTS communication_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            channel_id TEXT NOT NULL,
            sender_id TEXT NOT NULL, -- agent_id
            event_type TEXT NOT NULL, -- message, mention, reaction, file_share
            content TEXT, -- Message content or event details
            timestamp TIMESTAMP NOT NULL,
            thread_id TEXT DEFAULT NULL, -- For threaded conversations
            mentions TEXT DEFAULT NULL, -- JSON: [mentioned_agent_ids]
            sentiment_score REAL DEFAULT 0.0, -- -1.0 to 1.0
            urgency_level INTEGER DEFAULT 1, -- 1-5
            response_time_seconds INTEGER DEFAULT NULL,
            FOREIGN KEY (channel_id) REFERENCES communication_channels(channel_id)
        );
        
        CREATE TABLE IF NOT EXISTS calendar_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            calendar_source TEXT NOT NULL, -- google, outlook, internal
            title TEXT NOT NULL,
            description TEXT DEFAULT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP NOT NULL,
            timezone TEXT DEFAULT 'UTC',
            attendees TEXT, -- JSON: [agent_ids or emails]
            location TEXT DEFAULT NULL,
            meeting_type TEXT DEFAULT 'general', -- standup, review, planning
            project_id TEXT DEFAULT NULL,
            conflict_detected BOOLEAN DEFAULT FALSE,
            conflict_severity TEXT DEFAULT NULL, -- low, medium, high
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS collaboration_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight_type TEXT NOT NULL, -- bottleneck, communication_gap, overload
            affected_agents TEXT NOT NULL, -- JSON: [agent_ids]
            project_id TEXT DEFAULT NULL,
            severity TEXT DEFAULT 'medium', -- low, medium, high, critical
            description TEXT NOT NULL,
            suggested_actions TEXT, -- JSON: [action_items]
            confidence_score REAL DEFAULT 0.0, -- 0.0-1.0
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP DEFAULT NULL,
            resolution_notes TEXT DEFAULT NULL
        );
        """
        
        conn.executescript(collaboration_schema)
        return {"status": "collaboration_migrated", "tables": ["communication_channels", "communication_events", "calendar_events", "collaboration_insights"]}
    
    def _migrate_predictive(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Phase 7: Predictive Project Management"""
        print("🔮 Migrating Predictive Project Management...")
        
        predictive_schema = """
        -- Predictive Project Management (Phase 7)
        CREATE TABLE IF NOT EXISTS project_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            prediction_type TEXT NOT NULL, -- cost, timeline, quality, risk
            predicted_value REAL NOT NULL,
            confidence_interval_low REAL DEFAULT NULL,
            confidence_interval_high REAL DEFAULT NULL,
            confidence_score REAL NOT NULL, -- 0.0-1.0
            model_used TEXT NOT NULL, -- monte_carlo, ml_regression, etc.
            input_features TEXT, -- JSON: features used for prediction
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            target_date TIMESTAMP DEFAULT NULL, -- When prediction is for
            actual_value REAL DEFAULT NULL, -- Actual outcome (for learning)
            accuracy_score REAL DEFAULT NULL, -- How accurate was prediction
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        );
        
        CREATE TABLE IF NOT EXISTS resource_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_id TEXT UNIQUE NOT NULL,
            project_id TEXT DEFAULT NULL,
            resource_type TEXT NOT NULL, -- agent_hours, budget, equipment
            forecast_period_start DATE NOT NULL,
            forecast_period_end DATE NOT NULL,
            predicted_demand REAL NOT NULL,
            predicted_cost REAL DEFAULT NULL,
            availability_score REAL DEFAULT 1.0, -- 0.0-1.0
            risk_factors TEXT, -- JSON: [risk_descriptions]
            confidence_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS simulation_scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id TEXT UNIQUE NOT NULL,
            project_id TEXT NOT NULL,
            scenario_name TEXT NOT NULL,
            scenario_description TEXT,
            parameters_json TEXT NOT NULL, -- What-if parameters
            simulation_type TEXT DEFAULT 'monte_carlo', -- monte_carlo, discrete_event
            iterations INTEGER DEFAULT 1000,
            results_json TEXT, -- Simulation outcomes
            p50_outcome REAL DEFAULT NULL, -- Median result
            p90_outcome REAL DEFAULT NULL, -- 90th percentile
            risk_score REAL DEFAULT NULL, -- Overall risk assessment
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        );
        
        CREATE TABLE IF NOT EXISTS cost_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE NOT NULL,
            model_type TEXT NOT NULL, -- hourly_rate, fixed_price, value_based
            parameters_json TEXT NOT NULL, -- Model configuration
            applicable_roles TEXT, -- JSON: [roles this model applies to]
            baseline_cost REAL NOT NULL,
            variable_factors TEXT, -- JSON: factors affecting cost
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        conn.executescript(predictive_schema)
        return {"status": "predictive_migrated", "tables": ["project_predictions", "resource_forecasts", "simulation_scenarios", "cost_models"]}
    
    def _migrate_adaptive_learning(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Phase 8: Adaptive Learning & Self-Optimization"""
        print("🧠 Migrating Adaptive Learning...")
        
        learning_schema = """
        -- Adaptive Learning & Self-Optimization (Phase 8)
        CREATE TABLE IF NOT EXISTS learning_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            learning_type TEXT NOT NULL, -- online, batch, reinforcement
            model_name TEXT NOT NULL,
            data_source TEXT NOT NULL, -- team_outcomes, performance_data, etc.
            training_data_count INTEGER DEFAULT 0,
            learning_algorithm TEXT, -- sgd, random_forest, neural_net
            hyperparameters_json TEXT, -- Model configuration
            performance_before REAL DEFAULT NULL, -- Baseline metric
            performance_after REAL DEFAULT NULL, -- Post-training metric
            improvement_delta REAL DEFAULT NULL, -- Performance gain
            training_duration_seconds INTEGER DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP DEFAULT NULL,
            status TEXT DEFAULT 'running' -- running, completed, failed
        );
        
        CREATE TABLE IF NOT EXISTS model_performance_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            metric_name TEXT NOT NULL, -- accuracy, precision, recall, etc.
            metric_value REAL NOT NULL,
            data_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_sample_size INTEGER DEFAULT NULL,
            model_version TEXT DEFAULT 'latest',
            drift_detected BOOLEAN DEFAULT FALSE,
            drift_severity TEXT DEFAULT NULL, -- low, medium, high
            baseline_value REAL DEFAULT NULL -- For drift comparison
        );
        
        CREATE TABLE IF NOT EXISTS feedback_loops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loop_id TEXT UNIQUE NOT NULL,
            source_system TEXT NOT NULL, -- team_formation, analytics, etc.
            trigger_event TEXT NOT NULL, -- project_completion, weekly_review
            feedback_data_json TEXT NOT NULL, -- Actual feedback data
            target_model TEXT NOT NULL, -- Which model to improve
            processing_status TEXT DEFAULT 'pending', -- pending, processed, failed
            impact_score REAL DEFAULT NULL, -- How much this feedback matters
            processed_at TIMESTAMP DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS optimization_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            optimization_id TEXT UNIQUE NOT NULL,
            target_system TEXT NOT NULL, -- What was optimized
            optimization_type TEXT NOT NULL, -- hyperparameter, feature_selection
            parameters_before TEXT, -- JSON: before state
            parameters_after TEXT, -- JSON: after state
            performance_improvement REAL DEFAULT 0.0, -- % improvement
            validation_method TEXT, -- cross_validation, holdout, etc.
            rollback_available BOOLEAN DEFAULT TRUE,
            applied_to_production BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rollback_at TIMESTAMP DEFAULT NULL
        );
        """
        
        conn.executescript(learning_schema)
        return {"status": "learning_migrated", "tables": ["learning_sessions", "model_performance_tracking", "feedback_loops", "optimization_history"]}
    
    def _migrate_quantum_intelligence(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Phase 9: Quantum Intelligence & Predictive Evolution"""
        print("⚛️  Migrating Quantum Intelligence...")
        
        quantum_schema = """
        -- Quantum Intelligence & Predictive Evolution (Phase 9)
        CREATE TABLE IF NOT EXISTS quantum_problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id TEXT UNIQUE NOT NULL,
            problem_type TEXT NOT NULL, -- optimization, simulation, ml_enhancement
            problem_description TEXT,
            classical_formulation TEXT, -- Original problem statement
            quantum_formulation TEXT, -- QUBO/Ising model
            variables_count INTEGER DEFAULT 0,
            constraints_count INTEGER DEFAULT 0,
            problem_complexity TEXT DEFAULT 'medium', -- low, medium, high, exponential
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS quantum_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_id TEXT UNIQUE NOT NULL,
            problem_id TEXT NOT NULL,
            quantum_backend TEXT NOT NULL, -- simulator, ibmq_qasm, braket
            algorithm_used TEXT NOT NULL, -- qaoa, vqe, qml
            circuit_depth INTEGER DEFAULT NULL,
            qubit_count INTEGER DEFAULT NULL,
            execution_parameters_json TEXT, -- Algorithm-specific params
            execution_time_seconds REAL DEFAULT NULL,
            quantum_result_json TEXT, -- Raw quantum results
            classical_postprocessing TEXT, -- Any classical processing
            final_solution_json TEXT, -- Processed final answer
            solution_quality REAL DEFAULT NULL, -- 0.0-1.0
            confidence_score REAL DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP DEFAULT NULL,
            status TEXT DEFAULT 'pending', -- pending, running, completed, failed
            error_message TEXT DEFAULT NULL,
            FOREIGN KEY (problem_id) REFERENCES quantum_problems(problem_id)
        );
        
        CREATE TABLE IF NOT EXISTS quantum_vs_classical (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison_id TEXT UNIQUE NOT NULL,
            problem_id TEXT NOT NULL,
            classical_solution_json TEXT, -- Classical algorithm result
            quantum_solution_json TEXT, -- Quantum algorithm result
            classical_execution_time REAL, -- Seconds
            quantum_execution_time REAL, -- Seconds  
            solution_quality_classical REAL, -- 0.0-1.0
            solution_quality_quantum REAL, -- 0.0-1.0
            quantum_advantage BOOLEAN DEFAULT FALSE,
            advantage_factor REAL DEFAULT 1.0, -- How much better quantum was
            cost_comparison TEXT, -- JSON: cost analysis
            business_impact_json TEXT, -- Business value analysis
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES quantum_problems(problem_id)
        );
        
        CREATE TABLE IF NOT EXISTS evolutionary_adaptations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            adaptation_id TEXT UNIQUE NOT NULL,
            system_component TEXT NOT NULL, -- team_formation, analytics, etc.
            adaptation_type TEXT NOT NULL, -- algorithm_evolution, parameter_evolution
            generation_number INTEGER DEFAULT 1,
            parent_adaptations TEXT, -- JSON: previous adaptations
            genetic_operators TEXT, -- JSON: mutation, crossover details
            fitness_function TEXT, -- How performance is measured
            fitness_score REAL NOT NULL,
            performance_metrics_json TEXT, -- Detailed performance data
            survival_probability REAL DEFAULT 0.5, -- Likelihood to continue
            applied_to_system BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deprecated_at TIMESTAMP DEFAULT NULL
        );
        """
        
        conn.executescript(quantum_schema)
        return {"status": "quantum_migrated", "tables": ["quantum_problems", "quantum_executions", "quantum_vs_classical", "evolutionary_adaptations"]}
    
    def _create_performance_indexes(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Create database indexes for optimal performance"""
        print("🔍 Creating performance indexes...")
        
        indexes = """
        -- Performance Indexes for V2.0
        CREATE INDEX IF NOT EXISTS idx_team_history_project ON team_history(project_id);
        CREATE INDEX IF NOT EXISTS idx_team_history_success ON team_history(outcome_success DESC);
        
        CREATE INDEX IF NOT EXISTS idx_agent_perf_agent ON agent_performance(agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_perf_project ON agent_performance(project_id);
        CREATE INDEX IF NOT EXISTS idx_agent_perf_score ON agent_performance(individual_score DESC);
        
        CREATE INDEX IF NOT EXISTS idx_team_synergy_agents ON team_synergy(agent_a, agent_b);
        CREATE INDEX IF NOT EXISTS idx_team_synergy_score ON team_synergy(synergy_score DESC);
        
        CREATE INDEX IF NOT EXISTS idx_analytics_source ON analytics_dataset(source, category);
        CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics_dataset(sync_timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_reports_generated ON analytics_reports(generated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_reports_expires ON analytics_reports(expires_at);
        
        CREATE INDEX IF NOT EXISTS idx_comm_events_channel ON communication_events(channel_id);
        CREATE INDEX IF NOT EXISTS idx_comm_events_timestamp ON communication_events(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_comm_events_sender ON communication_events(sender_id);
        
        CREATE INDEX IF NOT EXISTS idx_calendar_events_time ON calendar_events(start_time, end_time);
        CREATE INDEX IF NOT EXISTS idx_calendar_events_attendees ON calendar_events(attendees);
        
        CREATE INDEX IF NOT EXISTS idx_predictions_project ON project_predictions(project_id);
        CREATE INDEX IF NOT EXISTS idx_predictions_type ON project_predictions(prediction_type);
        CREATE INDEX IF NOT EXISTS idx_predictions_date ON project_predictions(prediction_date DESC);
        
        CREATE INDEX IF NOT EXISTS idx_learning_sessions_model ON learning_sessions(model_name);
        CREATE INDEX IF NOT EXISTS idx_learning_sessions_completed ON learning_sessions(completed_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_model_tracking_model ON model_performance_tracking(model_name);
        CREATE INDEX IF NOT EXISTS idx_model_tracking_timestamp ON model_performance_tracking(data_timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_quantum_executions_problem ON quantum_executions(problem_id);
        CREATE INDEX IF NOT EXISTS idx_quantum_executions_status ON quantum_executions(status);
        CREATE INDEX IF NOT EXISTS idx_quantum_executions_completed ON quantum_executions(completed_at DESC);
        """
        
        conn.executescript(indexes)
        return {"status": "indexes_created", "count": 20}
    
    def _insert_sample_data(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Insert sample data for testing and demonstration"""
        print("📝 Inserting sample data...")
        
        # Sample agents
        sample_agents = [
            ("agent_001", "Alice Developer", "alice@example.com", '{"python": 0.95, "react": 0.8, "sql": 0.7}', "available", 120.0, "UTC+1"),
            ("agent_002", "Bob Designer", "bob@example.com", '{"figma": 0.9, "ui_ux": 0.95, "html_css": 0.8}', "available", 100.0, "UTC+1"),
            ("agent_003", "Charlie DevOps", "charlie@example.com", '{"docker": 0.9, "kubernetes": 0.8, "aws": 0.85}', "available", 130.0, "UTC"),
            ("agent_004", "Diana PM", "diana@example.com", '{"project_management": 0.95, "agile": 0.9, "stakeholder_management": 0.85}', "available", 110.0, "UTC+2"),
            ("agent_005", "Eve QA", "eve@example.com", '{"testing": 0.9, "automation": 0.8, "selenium": 0.75}', "available", 90.0, "UTC")
        ]
        
        for agent in sample_agents:
            conn.execute("""
            INSERT OR REPLACE INTO agents 
            (agent_id, name, email, skills, status, cost_per_hour, timezone)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, agent)
        
        # Sample projects
        sample_projects = [
            ("proj_001", "E-commerce Platform", "Build modern e-commerce platform", "active", 50000.0, "2024-12-31"),
            ("proj_002", "Mobile Banking App", "Secure mobile banking application", "planning", 75000.0, "2024-11-30"),
            ("proj_003", "AI Analytics Dashboard", "Machine learning analytics platform", "active", 40000.0, "2024-10-31")
        ]
        
        for project in sample_projects:
            conn.execute("""
            INSERT OR REPLACE INTO projects 
            (project_id, name, description, status, budget, deadline)
            VALUES (?, ?, ?, ?, ?, ?)
            """, project)
        
        # Sample team history
        team_history_samples = [
            ("proj_001", '["agent_001", "agent_002", "agent_003"]', "ai_recommended", 0.85, -0.05, 0.1, 0.9, 0.8, 0.85, '{"communication": "excellent", "technical_debt": "minimal"}'),
            ("proj_002", '["agent_001", "agent_004", "agent_005"]', "manual_selection", 0.75, 0.15, -0.05, 0.8, 0.7, 0.78, '{"timeline_pressure": "high", "scope_creep": "moderate"}')
        ]
        
        for history in team_history_samples:
            conn.execute("""
            INSERT OR REPLACE INTO team_history 
            (project_id, team_composition, formation_strategy, outcome_success, budget_delta, timeline_delta, quality_score, team_satisfaction, collaboration_rating, lessons_learned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, history)
        
        # Sample KPI definitions
        kpi_definitions = [
            ("team_success_rate", "Percentage of projects completed successfully", "AVG(outcome_success) FROM team_history", 0.85, "%", "performance", "daily", True),
            ("avg_budget_variance", "Average budget variance across projects", "AVG(ABS(budget_delta)) FROM team_history", 0.10, "%", "financial", "weekly", True),
            ("agent_utilization", "Average agent utilization rate", "COUNT(DISTINCT agent_id) / 5.0 FROM agent_performance", 0.80, "%", "performance", "daily", True),
            ("collaboration_score", "Average team collaboration rating", "AVG(collaboration_rating) FROM team_history", 0.85, "rating", "quality", "weekly", True)
        ]
        
        for kpi in kpi_definitions:
            conn.execute("""
            INSERT OR REPLACE INTO kpi_definitions 
            (kpi_name, kpi_description, calculation_formula, target_value, unit, category, update_frequency, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, kpi)
        
        conn.commit()
        return {"status": "sample_data_inserted", "agents": len(sample_agents), "projects": len(sample_projects)}
    
    def _verify_migration(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Verify migration completed successfully"""
        print("✅ Verifying migration...")
        
        # Count tables
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        
        # Count indexes  
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
        index_count = cursor.fetchone()[0]
        
        # Check sample data
        cursor.execute("SELECT COUNT(*) FROM agents")
        agent_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM kpi_definitions")
        kpi_count = cursor.fetchone()[0]
        
        # Test complex query (join across V2 tables)
        cursor.execute("""
        SELECT COUNT(*) 
        FROM team_history th 
        JOIN agent_performance ap ON th.project_id = ap.project_id
        """)
        join_test = cursor.fetchone()[0]
        
        verification_results = {
            "total_tables": table_count,
            "total_indexes": index_count,
            "sample_agents": agent_count,
            "sample_kpis": kpi_count,
            "cross_table_joins": join_test,
            "migration_successful": table_count >= 20 and agent_count >= 5
        }
        
        return verification_results

def main():
    """Execute Migration"""
    print("🚀 Agent Zero V2.0 Database Migration")
    print("=" * 50)
    
    migrator = AgentZeroDatabaseMigrator()
    
    try:
        results = migrator.run_migration()
        
        print("\n📊 Migration Summary:")
        print("=" * 30)
        
        for phase, result in results.items():
            if isinstance(result, dict) and "status" in result:
                print(f"✅ {phase}: {result['status']}")
                if "tables" in result:
                    print(f"   Tables: {', '.join(result['tables'])}")
            else:
                print(f"📝 {phase}: {result}")
        
        if results.get("verification", {}).get("migration_successful", False):
            print("\n🎉 Migration completed successfully!")
            print(f"📊 Total tables: {results['verification']['total_tables']}")
            print(f"🔍 Total indexes: {results['verification']['total_indexes']}")
            print(f"👥 Sample agents: {results['verification']['sample_agents']}")
        else:
            print("\n⚠️  Migration completed with warnings")
            
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())