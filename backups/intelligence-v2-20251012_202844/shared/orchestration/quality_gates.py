"""
Quality Gates Manager
System punktów kontroli jakości w projekcie
"""

from enum import Enum
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .task_decomposer import Task, TaskStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Status quality gate"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateSeverity(Enum):
    """Ważność quality gate"""
    CRITICAL = "critical"  # Musi przejść, blokuje deployment
    HIGH = "high"          # Powinien przejść, wymaga approval
    MEDIUM = "medium"      # Warto przejść, ostrzeżenie
    LOW = "low"           # Informacyjny


@dataclass
class QualityGate:
    """Punkt kontroli jakości"""
    gate_id: str
    name: str
    description: str
    severity: GateSeverity
    criteria: List[str]
    required_for_deployment: bool = True
    
    # Status
    status: GateStatus = GateStatus.PENDING
    checked_at: Optional[datetime] = None
    passed_criteria: List[str] = field(default_factory=list)
    failed_criteria: List[str] = field(default_factory=list)
    
    # Human approval
    requires_human_approval: bool = False
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    
    # Results
    result_details: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    
    def check_criterion(self, criterion: str, passed: bool, details: Optional[str] = None):
        """Sprawdź pojedyncze kryterium"""
        if passed:
            self.passed_criteria.append(criterion)
        else:
            self.failed_criteria.append(criterion)
            if details:
                self.result_details[criterion] = details
    
    def is_passed(self) -> bool:
        """Czy gate przeszedł wszystkie kryteria"""
        total = len(self.criteria)
        passed = len(self.passed_criteria)
        
        return passed == total and self.status == GateStatus.PASSED
    
    def get_pass_rate(self) -> float:
        """Procent przejściowych kryteriów"""
        if not self.criteria:
            return 1.0
        return len(self.passed_criteria) / len(self.criteria)


class QualityGateManager:
    """
    Manager quality gates
    Definiuje, wykonuje i monitoruje punkty kontroli jakości
    """
    
    def __init__(self):
        self.gates: Dict[str, QualityGate] = {}
        self.gate_order: List[str] = []
        self.checks: Dict[str, Callable] = {}
        logger.info("QualityGateManager zainicjalizowany")
    
    def register_gate(
        self,
        gate: QualityGate,
        check_function: Optional[Callable] = None
    ):
        """
        Zarejestruj quality gate
        
        Args:
            gate: QualityGate object
            check_function: Opcjonalna funkcja sprawdzająca (gate, context) -> bool
        """
        self.gates[gate.gate_id] = gate
        self.gate_order.append(gate.gate_id)
        
        if check_function:
            self.checks[gate.gate_id] = check_function
        
        logger.info(f"Zarejestrowano gate: {gate.name} ({gate.severity.value})")
    
    def define_standard_gates(self):
        """Zdefiniuj standardowe quality gates dla projektów"""
        
        # Gate 1: Architecture Review
        self.register_gate(
            QualityGate(
                gate_id="arch_review",
                name="Architecture Review",
                description="Przegląd architektury systemu",
                severity=GateSeverity.CRITICAL,
                criteria=[
                    "Diagram C4 Level 1-2 obecny",
                    "ADR dla kluczowych decyzji",
                    "Security considerations udokumentowane",
                    "Scalability analysis wykonany"
                ],
                requires_human_approval=True
            )
        )
        
        # Gate 2: Code Quality
        self.register_gate(
            QualityGate(
                gate_id="code_quality",
                name="Code Quality Gate",
                description="Standardy jakości kodu",
                severity=GateSeverity.CRITICAL,
                criteria=[
                    "Test coverage >= 80%",
                    "No critical code smells",
                    "Type hints complete (Python)",
                    "Linting passed (0 errors)",
                    "Code review completed"
                ]
            )
        )
        
        # Gate 3: Security Audit
        self.register_gate(
            QualityGate(
                gate_id="security_audit",
                name="Security Audit",
                description="Audyt bezpieczeństwa",
                severity=GateSeverity.CRITICAL,
                criteria=[
                    "No critical vulnerabilities",
                    "No high vulnerabilities",
                    "OWASP Top 10 checked",
                    "Dependencies scanned",
                    "Secrets not in code"
                ],
                requires_human_approval=True
            )
        )
        
        # Gate 4: Performance Benchmarks
        self.register_gate(
            QualityGate(
                gate_id="performance",
                name="Performance Benchmarks",
                description="Testy wydajnościowe",
                severity=GateSeverity.HIGH,
                criteria=[
                    "API response time < 200ms (p95)",
                    "Database queries < 100ms",
                    "Memory usage stable",
                    "Load test passed (1000 RPS)"
                ]
            )
        )
        
        # Gate 5: Integration Tests
        self.register_gate(
            QualityGate(
                gate_id="integration_tests",
                name="Integration Tests",
                description="Testy integracyjne",
                severity=GateSeverity.HIGH,
                criteria=[
                    "All integration tests passing",
                    "E2E smoke tests passing",
                    "API contract tests passing"
                ]
            )
        )
        
        # Gate 6: Documentation
        self.register_gate(
            QualityGate(
                gate_id="documentation",
                name="Documentation Gate",
                description="Kompletność dokumentacji",
                severity=GateSeverity.MEDIUM,
                criteria=[
                    "README.md complete",
                    "API documentation present",
                    "Installation guide present",
                    "Architecture documented"
                ],
                required_for_deployment=False
            )
        )
        
        # Gate 7: Deployment Readiness
        self.register_gate(
            QualityGate(
                gate_id="deployment_ready",
                name="Deployment Readiness",
                description="Gotowość do wdrożenia",
                severity=GateSeverity.CRITICAL,
                criteria=[
                    "Docker images built",
                    "CI/CD pipeline working",
                    "Environment config ready",
                    "Rollback plan documented",
                    "Monitoring configured"
                ],
                requires_human_approval=True
            )
        )
        
        logger.info(f"Zdefiniowano {len(self.gates)} standardowych gates")
    
    def check_gate(
        self,
        gate_id: str,
        context: Dict[str, Any]
    ) -> GateStatus:
        """
        Sprawdź quality gate
        
        Args:
            gate_id: ID gate do sprawdzenia
            context: Kontekst z danymi do sprawdzenia
        
        Returns:
            GateStatus
        """
        if gate_id not in self.gates:
            logger.error(f"Gate {gate_id} nie istnieje")
            return GateStatus.FAILED
        
        gate = self.gates[gate_id]
        gate.status = GateStatus.IN_PROGRESS
        gate.checked_at = datetime.now()
        
        # Jeśli jest custom check function, użyj jej
        if gate_id in self.checks:
            try:
                result = self.checks[gate_id](gate, context)
                gate.status = GateStatus.PASSED if result else GateStatus.FAILED
            except Exception as e:
                logger.error(f"Błąd sprawdzania gate {gate_id}: {e}")
                gate.status = GateStatus.FAILED
                gate.notes = str(e)
        else:
            # Domyślnie - manual check przez kryteria
            logger.info(f"Gate {gate_id} wymaga manualnego sprawdzenia")
        
        logger.info(
            f"Gate {gate.name}: {gate.status.value} "
            f"({len(gate.passed_criteria)}/{len(gate.criteria)} kryteriów)"
        )
        
        return gate.status
    
    def approve_gate(
        self,
        gate_id: str,
        approver: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Human approval dla gate
        
        Args:
            gate_id: ID gate
            approver: Kto zatwierdził
            notes: Opcjonalne notatki
        
        Returns:
            True jeśli zatwierdzono
        """
        if gate_id not in self.gates:
            return False
        
        gate = self.gates[gate_id]
        
        if not gate.requires_human_approval:
            logger.warning(f"Gate {gate_id} nie wymaga human approval")
            return False
        
        gate.approved_by = approver
        gate.approval_timestamp = datetime.now()
        gate.notes = notes
        gate.status = GateStatus.PASSED
        
        logger.info(f"✅ Gate {gate.name} zatwierdzony przez {approver}")
        return True
    
    def check_all_gates(
        self,
        context: Dict[str, Any],
        stop_on_failure: bool = False
    ) -> Dict[str, GateStatus]:
        """
        Sprawdź wszystkie gates w kolejności
        
        Args:
            context: Kontekst projektu
            stop_on_failure: Czy zatrzymać na pierwszym błędzie
        
        Returns:
            Dict {gate_id: GateStatus}
        """
        results = {}
        
        for gate_id in self.gate_order:
            status = self.check_gate(gate_id, context)
            results[gate_id] = status
            
            if stop_on_failure and status == GateStatus.FAILED:
                gate = self.gates[gate_id]
                if gate.severity in [GateSeverity.CRITICAL, GateSeverity.HIGH]:
                    logger.error(f"Critical gate {gate.name} failed - zatrzymano")
                    break
        
        return results
    
    def get_deployment_blockers(self) -> List[QualityGate]:
        """Zwróć gates które blokują deployment"""
        blockers = []
        
        for gate in self.gates.values():
            if (gate.required_for_deployment and 
                gate.status != GateStatus.PASSED and
                gate.severity in [GateSeverity.CRITICAL, GateSeverity.HIGH]):
                blockers.append(gate)
        
        if blockers:
            logger.warning(f"⚠️  {len(blockers)} gates blokuje deployment")
            for gate in blockers:
                logger.warning(f"   - {gate.name}: {gate.status.value}")
        
        return blockers
    
    def is_ready_for_deployment(self) -> bool:
        """Czy projekt jest gotowy do deployment"""
        blockers = self.get_deployment_blockers()
        ready = len(blockers) == 0
        
        if ready:
            logger.info("✅ Wszystkie quality gates PASSED - gotowy do deployment")
        else:
            logger.warning(f"❌ {len(blockers)} gates blokuje deployment")
        
        return ready
    
    def get_gate_summary(self) -> Dict[str, Any]:
        """Zwróć podsumowanie wszystkich gates"""
        total = len(self.gates)
        passed = sum(1 for g in self.gates.values() if g.status == GateStatus.PASSED)
        failed = sum(1 for g in self.gates.values() if g.status == GateStatus.FAILED)
        pending = sum(1 for g in self.gates.values() if g.status == GateStatus.PENDING)
        
        return {
            'total_gates': total,
            'passed': passed,
            'failed': failed,
            'pending': pending,
            'pass_rate': passed / total if total > 0 else 0.0,
            'deployment_ready': self.is_ready_for_deployment(),
            'gates': [
                {
                    'id': g.gate_id,
                    'name': g.name,
                    'status': g.status.value,
                    'severity': g.severity.value,
                    'pass_rate': g.get_pass_rate(),
                    'requires_approval': g.requires_human_approval,
                    'approved_by': g.approved_by
                }
                for g in self.gates.values()
            ]
        }
    
    def reset_all_gates(self):
        """Zresetuj wszystkie gates (np. na nową wersję)"""
        for gate in self.gates.values():
            gate.status = GateStatus.PENDING
            gate.passed_criteria.clear()
            gate.failed_criteria.clear()
            gate.checked_at = None
            gate.approved_by = None
            gate.approval_timestamp = None
            gate.result_details.clear()
        
        logger.info("Zresetowano wszystkie quality gates")


def create_quality_gate_manager() -> QualityGateManager:
    """Utwórz QualityGateManager z standardowymi gates"""
    manager = QualityGateManager()
    manager.define_standard_gates()
    return manager
