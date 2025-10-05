"""
Response Parser
Parse and extract structured data from LLM responses
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Parse LLM responses and extract structured data
    """
    
    @staticmethod
    def extract_code_blocks(response: str, language: str = None) -> List[Dict[str, str]]:
        """
        Extract code blocks from markdown response
        
        Args:
            response: LLM response text
            language: Specific language to extract (optional)
        
        Returns:
            List of dicts with 'language' and 'code'
        """
        pattern = r'``````'
        matches = re.findall(pattern, response, re.DOTALL)
        
        code_blocks = []
        for lang, code in matches:
            if language is None or lang == language:
                code_blocks.append({
                    'language': lang or 'text',
                    'code': code.strip()
                })
        
        logger.info(f"Extracted {len(code_blocks)} code blocks")
        return code_blocks
    
    @staticmethod
    def extract_first_code_block(response: str, language: str = None) -> Optional[str]:
        """Extract first code block"""
        blocks = ResponseParser.extract_code_blocks(response, language)
        return blocks[0]['code'] if blocks else None
    
    @staticmethod
    def parse_code_review(response: str) -> List[Dict[str, Any]]:
        """
        Parse code review response
        
        Expected format:
        **Severity**: Critical
        **Issue**: Description
        **Location**: Line 10-15
        **Fix**: Suggested fix
        
        Returns:
            List of review issues
        """
        issues = []
        
        sections = re.split(r'\*\*Severity\*\*:', response)
        
        for section in sections[1:]:
            try:
                severity_match = re.search(r'(Critical|High|Medium|Low)', section)
                issue_match = re.search(r'\*\*Issue\*\*:\s*(.+?)(?=\*\*|$)', section, re.DOTALL)
                location_match = re.search(r'\*\*Location\*\*:\s*(.+?)(?=\*\*|$)', section)
                fix_match = re.search(r'\*\*Fix\*\*:\s*(.+?)(?=\*\*|$)', section, re.DOTALL)
                
                if severity_match and issue_match:
                    issues.append({
                        'severity': severity_match.group(1),
                        'issue': issue_match.group(1).strip(),
                        'location': location_match.group(1).strip() if location_match else None,
                        'fix': fix_match.group(1).strip() if fix_match else None
                    })
            except Exception as e:
                logger.warning(f"Failed to parse review section: {e}")
                continue
        
        logger.info(f"Parsed {len(issues)} code review issues")
        return issues
    
    @staticmethod
    def parse_numbered_list(response: str, section_title: str = None) -> List[str]:
        """
        Parse numbered list from response
        
        Args:
            response: LLM response
            section_title: Optional section title to extract from
        
        Returns:
            List of items
        """
        if section_title:
            pattern = rf'{section_title}[:\n]+(.*?)(?=\n#|\Z)'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                response = match.group(1)
        
        pattern = r'^\s*\d+[\.\)]\s+(.+?)(?=\n\s*\d+[\.\)]|\Z)'
        matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
        
        items = [m.strip() for m in matches]
        logger.info(f"Parsed {len(items)} list items")
        return items
    
    @staticmethod
    def parse_bullet_list(response: str) -> List[str]:
        """Parse bullet list (- or *)"""
        pattern = r'^\s*[\-\*]\s+(.+?)$'
        matches = re.findall(pattern, response, re.MULTILINE)
        return [m.strip() for m in matches]
    
    @staticmethod
    def parse_post_mortem(response: str) -> Dict[str, List[str]]:
        """
        Parse post-mortem analysis
        
        Returns:
            Dict with: what_went_well, what_went_wrong, lessons_learned, recommendations
        """
        result = {
            'what_went_well': [],
            'what_went_wrong': [],
            'lessons_learned': [],
            'recommendations': []
        }
        
        sections = {
            'what_went_well': r'(?:Co poszło dobrze|What went well)[:\n]+(.*?)(?=\n#|Co poszło|What went|\Z)',
            'what_went_wrong': r'(?:Co poszło źle|What went wrong)[:\n]+(.*?)(?=\n#|Lessons|Rekomendacje|\Z)',
            'lessons_learned': r'(?:Lessons learned|Wnioski)[:\n]+(.*?)(?=\n#|Rekomendacje|\Z)',
            'recommendations': r'(?:Rekomendacje|Recommendations)[:\n]+(.*?)(?=\n#|\Z)'
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)
                items = ResponseParser.parse_numbered_list(text)
                if not items:
                    items = ResponseParser.parse_bullet_list(text)
                result[key] = items
        
        logger.info(f"Parsed post-mortem: {len(result['lessons_learned'])} lessons")
        return result
    
    @staticmethod
    def clean_response(response: str) -> str:
        """Clean response - remove excessive whitespace"""
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = response.strip()
        return response
    
    @staticmethod
    def extract_json_block(response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON block from response"""
        pattern = r'``````'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON block")
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return None
