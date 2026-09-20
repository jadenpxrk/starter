"""Bounded request-local suffix proposals with exact full-model verification.

History lookup NEVER supplies an answer. Only target-model argmax predictions
on an unbroken accepted prefix may be emitted. No prompt/output data survives
one generate() call. All batching remains in original sequence order.
"""

from collections import deque

BLOCK = 4
DRAFT = BLOCK - 1
MIN_N, MAX_N = 3, 6
WINDOW = 4096  # Bounds proposal-search work only, NOT model context or KV.
MAX_HISTORY_TOKENS = 65536  # CPU index budget across the batch.
MAX_UNPRODUCTIVE = 4  # Stop after four verifications accepting zero proposals.


class SuffixDraft:
    def __init__(self, tokens, window=WINDOW):
        if window < MAX_N + DRAFT:
            raise ValueError("proposal window is too small")
        self.window = window
        self.tokens, self.index, self.expiry = [], {}, deque()
        for token in tokens[-window:]:
            self.append(token)

    def append(self, token):
        self.tokens.append(int(token))
        if len(self.tokens) > 2 * self.window:
            tail = self.tokens[-self.window:]
            self.tokens, self.index, self.expiry = [], {}, deque()
            for value in tail:
                self.append(value)
            return
        end = len(self.tokens) - DRAFT
        for n in range(MIN_N, MAX_N + 1):
            if end >= n:
                key = tuple(self.tokens[end - n:end])
                self.index[key] = end
                self.expiry.append((end, key))
        cutoff = len(self.tokens) - self.window
        while self.expiry and self.expiry[0][0] < cutoff:
            old, key = self.expiry.popleft()
            if self.index.get(key) == old:
                del self.index[key]

    def propose(self):
        for n in range(MAX_N, MIN_N - 1, -1):
            if len(self.tokens) >= n:
                end = self.index.get(tuple(self.tokens[-n:]))
                if end is not None:
                    # The index is inserted only after ALL successors exist.
                    return self.tokens[end:end + DRAFT].copy()
        return None


def accepted_count(inputs, predictions):
    """Synchronized batch prefix: a accepted drafts plus one target prediction.

    inputs[:,0] is already known. To accept inputs[:,j], its ID must equal
    predictions[:,j-1] in EVERY lane. After the first mismatch, later predictions
    are on an unaccepted prefix and MUST NOT be emitted.
    """
    if (not inputs or len(inputs) != len(predictions)
            or any(len(row) != BLOCK for row in inputs + predictions)):
        raise ValueError("expected matching nonempty [batch,BLOCK] integer lists")
    count = 1
    for j in range(1, BLOCK):
        if any(row[j] != pred[j - 1] for row, pred in zip(inputs, predictions)):
            break
        count += 1
    return count


class LookupPlan:
    def __init__(self, state, verifier):
        self.state, self.verifier = state, verifier

    def capture(self):
        self.verifier.capture()

    def _step(self):
        if self.state.graph is None:
            self.state.step()
        else:
            self.state.graph.replay()

    def _current(self, advance=False):
        s = self.state
        s.host.copy_(s.tokens, non_blocking=True)
        if s.ready is not None:
            s.ready.record()
        if advance:
            self._step()
        if s.ready is not None:
            s.ready.synchronize()
        return s.host[:, 0].tolist()

    def emit(self, input_ids, count):
        if count <= 0:
            return
        # First output/next-step ordering is identical to #12. Building the
        # request-local lookup index happens AFTER the first yield.
        first = self._current(advance=count > 1)
        yield first
        if count == 1:
            return
        window = min(WINDOW, MAX_HISTORY_TOKENS // len(input_ids))
        history = [SuffixDraft(row, window=window) for row in input_ids]
        for h, value in zip(history, first):
            h.append(value)
        emitted, unproductive, pending = 1, 0, None
        while emitted < count:
            if pending is None:
                rows = [self._current()]
            else:
                predicted = self.verifier.finish()
                n = accepted_count(pending, predicted)
                self.verifier.commit(n)
                unproductive += int(n == 1)
                rows = [[predicted[b][i] for b in range(len(history))] for i in range(n)]
                pending = None
            for i, row in enumerate(rows):
                emitted += 1
                for h, value in zip(history, row):
                    h.append(value)
                if i + 1 < len(rows):
                    yield row
                    continue
                # Only the last buffered, VERIFIED output is the state frontier.
                # Delegate tails/poor acceptance to the unmodified #12 emitter;
                # it includes this not-yet-yielded frontier token exactly once.
                remaining = count - emitted
                if remaining < BLOCK or unproductive >= MAX_UNPRODUCTIVE:
                    yield from self.state.emit(remaining + 1)
                    return
                proposals = [h.propose() for h in history]
                if all(p is not None for p in proposals):
                    pending = [[row[b], *p] for b, p in enumerate(proposals)]
                    self.verifier.enqueue(pending)
                else:
                    self._step()
                yield row


def make_lookup_plan(state):
    # No public-shape table. The existing validated prefill/Flash eligibility
    # checks establish this model/layout; short outputs use #12 unchanged.
    if (state.prefill_plan is None or state.flash_context is None
            or state.fused_forward is None or state.shape[2] < BLOCK + 2
            or state.shape[0] > MAX_HISTORY_TOKENS // (MAX_N + DRAFT)):
        return None
    from verify import VerificationPlan
    return LookupPlan(state, VerificationPlan(state, BLOCK))
