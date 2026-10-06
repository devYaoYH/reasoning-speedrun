"""V1 integer prospective-answer extraction. Correctness belongs to the grader."""

import re


class CandidateDetector:
    """Closed integer boxes and complete answer clauses, independently per channel.

    In particular, a network boundary after one digit never completes an answer.
    Markers remain prospective; correctness comes exclusively from the oracle.
    """

    box = re.compile("\\\\boxed\\s*\\{\\s*(\\d{1,3})\\s*\\}")
    line = re.compile(
        "(?i)^\\s*(?:\\*\\*)?Answer\\s*:\\s*\\$?\\s*(\\d{1,3})\\s*\\$?\\s*(?:\\*\\*)?\\s*[.]?\\s*$"
    )
    prose = re.compile(
        "(?i)\\b(?:final\\s+)?answer\\s+(?:is|would\\s+be|should\\s+be|must\\s+be|might\\s+be)\\s+\\$?\\s*(\\d{1,3})\\s*\\$?(?=\\s*(?:[.。!?;,](?:\\s|$)|$))"
    )

    def __init__(self):
        self.text = {"content": "", "reasoning": ""}
        self.scanned = {"content": 0, "reasoning": 0}
        self.line_start = {"content": 0, "reasoning": 0}

    def feed(self, part, delta, eof=False):
        self.text[part] += delta
        text = self.text[part]
        found = []
        for match in self.box.finditer(text, self.scanned[part]):
            found.append(
                {
                    "answer": int(match[1]),
                    "part": part,
                    "kind": "boxed",
                    "end": match.end(),
                }
            )
            self.scanned[part] = match.end()
        tail = text.rfind("\\boxed", self.scanned[part])
        self.scanned[part] = (
            tail if tail >= 0 else max(self.scanned[part], len(text) - 6)
        )
        start = self.line_start[part]
        while "\n" in text[start:] or (eof and start < len(text)):
            end = text.find("\n", start)
            end = len(text) if end < 0 else end + 1
            complete_line = text[start:end]
            match = self.line.fullmatch(complete_line)
            if match:
                found.append(
                    {
                        "answer": int(match[1]),
                        "part": part,
                        "kind": "answer_line",
                        "end": end,
                    }
                )
            for match in self.prose.finditer(complete_line):
                found.append(
                    {
                        "answer": int(match[1]),
                        "part": part,
                        "kind": "literal_prose",
                        "end": start + match.end(),
                        "line": complete_line.strip(),
                    }
                )
            start = end
        self.line_start[part] = start
        return found
