import json, re, logging
class TaskDecomposer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.strats = [
            self._direct, self._markdown, self._regex,
            self._line, self._adv_regex, self._fallback
        ]
    def parse(self, resp:str):
        for i,s in enumerate(self.strats,1):
            try:
                r = s(resp)
                if r:
                    logging.info(f"Parsed by strat {i}")
                    return r
            except:
                pass
        logging.warning("All strat failed")
        return self._fallback(resp)
    def _direct(self, r):
        j = r.strip()
        return json.loads(j) if j.startswith('{') and j.endswith('}') else None
    def _markdown(self, r):
        m = re.findall(r'``````', r, re.S)
        for j in m:
            try: return json.loads(j)
            except: pass
    def _regex(self, r):
        m = re.findall(r'\{(?:[^{}]|(?R))*\}', r)
        for j in m:
            try: return json.loads(j)
            except: pass
    def _line(self, r):
        lines, buf, cnt, on = r.split('\n'), [], 0, False
        for l in lines:
            if '{' in l and not on:
                on = True
            if on:
                buf.append(l)
                cnt += l.count('{') - l.count('}')
                if cnt <= 0:
                    break
        if buf:
            return json.loads('\n'.join(buf))
    def _adv_regex(self, r):
        return self._regex(r)
    def _fallback(self, r):
        return {'task':'fallback','original':r[:200]+'...'}
if __name__=='__main__':
    td = TaskDecomposer()
    for t in ['{"a":1}', '``````', 'no json here']:
        print(td.parse(t))
