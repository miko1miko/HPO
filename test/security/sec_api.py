import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PurpleLlama.CodeShield.codeshield import CodeShield, Language
import asyncio
from typing import Tuple, Dict, List

async def security_check(code: str, language: str) -> Tuple[bool, str, List[Dict]]:
    language_map = {
        'c': Language.C,
        'cpp': Language.CPP,
        'java': Language.JAVA,
        'javascript': Language.JAVASCRIPT,
        'python': Language.PYTHON,
        'rust': Language.RUST
    }
    
    if language.lower() not in language_map:
        raise ValueError(f"Unsupported language: {language}. Supported languages are: {list(language_map.keys())}")
    
    result = await CodeShield.scan_code(code, language=language_map[language.lower()])
    return result.is_insecure, result.recommended_treatment, result.issues_found

async def test_security_check():
    test_cases = [
        {
            'language': 'c',
            'code': """
#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n\nint main() {\n    int n, i, j, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8;\n    char command[100];\n    scanf(\"%d\", &n);\n    for (i = 0; i < n; i++) {\n        scanf(\"%d %d %d %d\", &x1, &y1, &x2, &y2);\n        for (j = i + 1; j < n; j++) {\n            scanf(\"%d %d %d %d\", &x3, &y3, &x4, &y4);\n            if (x1 == x3 && x2 == x4 && y1 == y3 && y2 == y4) {\n                scanf(\"%d %d %d %d\", &x5, &y5, &x6, &y6);\n                if (x1 == x5 && x2 == x6 && y1 == y5 && y2 == y6) {\n                    scanf(\"%d %d %d %d\", &x7, &y7, &x8, &y8);\n                    if (x1 == x7 && x2 == x8 && y1 == y7 && y2 == y8) {\n                        printf(\"YES\\n\");\n                        sprintf(command, \"log_result YES\");\n                        system(command);\n                        return 0;\n                    }\n                }\n            }\n        }\n    }\n    printf(\"NO\\n\");\n    sprintf(command, \"log_result NO\");\n    system(command);\n    return 0;\n}\n   
"""
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['language'].upper()} code:")
        print("Code snippet:")
        print(test_case['code'])
        
        try:
            is_insecure, treatment, issues = await security_check(
                test_case['code'],
                test_case['language']
            )
            
            print("\nResults:")
            print(f"Security issues found: {'Yes' if is_insecure else 'No'}")
            print(f"Recommended treatment: {treatment}")
            print("Issues found:")
            for issue in issues:
                print(f"- {issue}")
        except Exception as e:
            print(f"Error during security check: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_security_check())