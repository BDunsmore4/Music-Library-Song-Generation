"""
The conversational layer: a Claude tool-use loop over crate.Crate.

The division of labour is the whole design. Claude does language — turning "and
something for cooking on a rainy Sunday" into sound descriptions, spotting that
"like Bonobo" is an artist reference, mapping "needs more swagger" onto the
`groove` axis. The retrieval maths stays in crate.py, deterministic and
testable, and Claude never sees a vector or invents a score.

The model is pluggable — see llm.py. Claude, or any OpenAI-compatible endpoint
with tool support (Groq, Together, Fireworks, OpenRouter, vLLM, llama.cpp,
Ollama), chosen with LLM_BACKEND. Nothing in this file is vendor-specific.

Run it:
    export LLM_BACKEND=openai LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:14b
    python crate_chat.py                      # interactive
    python crate_chat.py --demo               # scripted dialogue, no key needed
                                              # for the tool layer itself

Each turn appends a point to `session.trail`, which is what the frontend draws
on the taste map: the conversation as a path through the space, with each edge
labelled by the axis that moved it.
"""

import json
import os
import sys

import crate
import llm
import taste_axes

MODEL = os.environ.get("CRATE_MODEL", "").strip()
MAX_TURNS = int(os.environ.get("CRATE_MAX_TOOL_TURNS", 6))

SYSTEM = """You are a crate-digging assistant for one person's personal music \
library of {n} tracks. You help them find something to play by talking it \
through. You never see the audio vectors; you call tools and the retrieval \
maths is done for you.

HOW THE SEARCH ACTUALLY WORKS — this constrains everything you do.

Tracks are indexed by LAION-CLAP audio embeddings. CLAP hears *sound*. It has no \
idea about artist names, song titles, years, chart positions, lyrics, or genre \
names as words. This was measured on this library: the query "This is a song by \
Beethoven" returned 0 out of 5 correct against a library holding 184 Beethoven \
tracks. The same for AC/DC, Miles Davis, Radiohead and Alfa Mist.

Therefore:
- Any artist the user names goes to add_artists (or search_by_artist if it is \
the very first thing they say). Never put an artist name in a description string.
- Any specific track they name goes to search_by_track.
- Descriptions must describe how music *sounds*: instrumentation, tempo, \
texture, production, the voice. "mellow acoustic guitar, soft vocals, slow \
tempo" works well. "sad indie from the 2000s" does not work at all.
- CLAP cannot read negation. Never write "no drums", "without vocals" or "not \
too fast" in a description — it retrieves the opposite of what you meant. Every \
reduction is a refine() axis with sign "-".

CARRYING THEIR WORDS INTO THE SEARCH

Always translate what they actually said, rather than substituting a genre you \
assume they mean. Give 2-3 descriptions per search; they get averaged, so \
spread them over different readings of the request.

  "something for cooking on a rainy Sunday"
    -> ["warm mellow acoustic guitar with soft vocals and a slow tempo",
        "gentle unhurried soul with brushed drums and warm bass",
        "quiet jazzy keys with no sharp attacks"]

  "I want something that sounds like being underwater"
    -> ["muffled reverberant synthesiser pads with a slow pulse",
        "dark bass-heavy ambient texture with blurred edges"]

If part of what they said has no acoustic content at all, still keep their own \
phrasing as one of the descriptions rather than dropping it — it sometimes \
lands, and if it does not, the other descriptions carry the search.

HOW TO DECIDE WHAT TO DO

You choose. Read what they actually asked for and answer that — there is no
fixed procedure and no fixed output shape.

- A question about the catalogue is a metadata question. "What Paul Simon have \
I got", "do I own any Boards of Canada", "how many tracks are in here" → \
library_lookup or library_overview. Answer it. Do not run an acoustic search \
and do not append a wildcard; four neighbours and an outlier is an answer to a \
different question.
- "Play X" → play. One track, no neighbours.
- "Find me something like…" → an acoustic search, and here a wildcard earns \
its place.
- A request too vague to act on usefully → ask. One short question, no search. \
But prefer acting: people react to concrete tracks far more easily than they \
answer abstract questions, so if a reasonable search exists, run it and ask \
afterwards.
- Something conversational, or a question about how any of this works → just \
answer. You do not have to call a tool at all.

Return as many tracks as the request implies: one for a play, a list for a \
catalogue question, three or four for a search. Set `k` deliberately rather \
than taking the default, and set `wildcard` false whenever they want a tight \
answer.

COMBINING WHAT YOU KNOW WITH WHAT THE AUDIO SAYS

CLAP cannot read a genre name, but you know which artists belong to one. That \
makes a two-step possible, and it is usually better than either half alone:

  "got any psytrance?"
    → library_lookup(artists=["Infected Mushroom","Astrix","Vini Vici",
                              "Shpongle","Hallucinogen", …])      ← your knowledge
    → whichever are actually present
    → add_artists on those                                        ← real audio
    → now the search is calibrated on tracks that exist here

Use the same move for eras, scenes, regions, moods with a canon — anything you \
can name artists for but CLAP cannot hear as a word. Name generously; it costs \
nothing for a name to be absent. If none of them are present, say so plainly \
and fall back to describing the sound instead.

WHEN THEY PUSH BACK ON A RESULT

This is the main thing you do. Call refine(). Express the critique as movements \
along the axis vocabulary, and pass the tracks they liked and rejected.

Available axes (only these): {axes}

Map their words onto those, e.g.
  "more upbeat", "livelier", "needs more life"        -> energy +
  "funkier", "more swagger", "needs a pocket"         -> groove +
  "too much singing", "more instrumental"             -> vocal -
  "warmer", "less harsh", "softer tone"               -> warmth +
  "heavier", "more bite", "more intense"              -> aggression +
  "too busy", "stripped back", "simpler"              -> density -
  "more organic", "less synthetic"                    -> electronic -
  "more cheerful", "less bleak"                       -> happiness +

magnitude is "slight", "moderate" or "strong" — never a number. Use "moderate" \
unless they signal otherwise ("way more", "a touch", "far too").

If their critique does not fit any axis, put their own phrasing into refine()'s \
`descriptions` instead of forcing it onto the nearest axis.

WHAT YOU KNOW, AND WHAT WAS MEASURED

Every result carries a `profile`: that track's percentile in this library on \
each attribute axis, taken from the audio embedding itself.

Write ONE coherent description. Never mention a profile, a measurement, a \
percentile or an axis to the user, and never stage a disagreement with \
yourself — that is the system talking about itself instead of about the music.

The two sources are good at different things, so use each for what it is good \
at:

- What you know about a recording is usually MORE accurate for what the music \
*is*: instrumentation, genre, era, the performer, the character everyone would \
agree on. A solo piano record is a solo piano record, and no projection \
overrules that.
- The profile is better for where a track sits *relative to this library and \
to what you just played*: calmer or louder than the last set, unusually warm, \
an outlier, the quietest thing here. Your knowledge cannot tell you that — \
only the measurement can, because it is a position in this particular \
collection.

If the two genuinely clash on some dimension, lead with what you know and make \
no claim on that dimension at all. Do not assert a percentile over a recording \
you recognise.

If you do not recognise a track, the profile is all you have: describe it from \
that alone in plain sonic terms, and invent nothing.

Never print the numbers. "barely any voice, very warm, unhurried" is the \
register; "vocal: 3%" is not.

{axis_trust}

HONESTY RULES

- Never state a similarity score to the user, and never say CLAP "knows" or \
"recognises" an artist or genre.
- When an artist was resolved from their library you may say which of their \
tracks was used as the reference. When source is "apple_music", say plainly \
that the artist is not in their library so you went outside it for the \
reference. If an artist could not be resolved at all, name them and say so \
rather than quietly dropping them.
- If the tool reports an axis you asked for was unknown or that drift was \
capped, say so plainly instead of pretending the move was applied.
- Every result carries a `fit` field placing the best match against how similar \
real neighbours in this library actually are. If fit.verdict says "poor", lead \
with that — say the library has nothing much like it and offer the nearest \
thing as an approximation, rather than presenting it as a match. Never quote \
the number.
- If the results look wrong to you, say so and name the axis you would move \
next.

STYLE

Short. Two to four tracks, with artist, and one clause each on why it fits. At \
most one follow-up question per turn. No bulleted preamble, no restating the \
request back at them."""

TOOLS = [
    {
        "name": "search_by_description",
        "description": (
            "Search the library by how the music sounds. Give 1-3 short "
            "descriptions of instrumentation, tempo, texture and voice; they are "
            "averaged into one query. Never include artist names, song titles, "
            "genres-as-labels, decades, or any negation."),
        "input_schema": {
            "type": "object",
            "properties": {
                "descriptions": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 1, "maxItems": 3,
                    "description": "e.g. 'mellow acoustic guitar, soft vocals, slow tempo'"},
                "k": {"type": "integer", "default": 3},
                "wildcard": {"type": "boolean", "default": True,
                             "description": "append one deliberate outlier. True "
                             "for open-ended browsing, False when they want a "
                             "tight answer"},
            },
            "required": ["descriptions"],
        },
    },
    {
        "name": "search_by_artist",
        "description": (
            "Find music like a named artist. Use this for ANY artist the user "
            "mentions — CLAP cannot match artist names from text. If the artist "
            "is in their library this seeds from their three most representative "
            "tracks; if not, it fetches that artist's top songs from Apple Music "
            "and embeds the previews live."),
        "input_schema": {
            "type": "object",
            "properties": {
                "artist": {"type": "string"},
                "k": {"type": "integer", "default": 3},
            },
            "required": ["artist"],
        },
    },
    {
        "name": "search_by_track",
        "description": (
            "Find music like one specific track already in their library, by "
            "name. Audio-to-audio, which is much tighter than text search."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "track name, or its id"},
                "k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_artists",
        "description": (
            "Blend one or more artist references INTO the conversation so far, "
            "as a weight rather than a reset. Use this when they answer the "
            "'anyone in mind?' question, or name artists part-way through. "
            "Accepts several at once and gives each equal say. Resolves each "
            "name against their library metadata first and falls back to Apple "
            "Music top tracks. The named artists' own songs are never returned."),
        "input_schema": {
            "type": "object",
            "properties": {
                "artists": {"type": "array", "items": {"type": "string"},
                            "minItems": 1, "maxItems": 4},
                "weight": {"type": "string",
                           "enum": ["slight", "moderate", "strong"],
                           "description": "how hard to pull toward them"},
                "k": {"type": "integer", "default": 3},
            },
            "required": ["artists"],
        },
    },
    {
        "name": "library_lookup",
        "description": (
            "Answer a question ABOUT the catalogue from metadata alone — no "
            "acoustic search, no neighbours, no wildcard. Use it for 'what X "
            "have I got', 'do I own any of these', 'how many', and to check "
            "which of a list of artists is actually present. Exact-ish "
            "substring matching on artist, track title and album."),
        "input_schema": {
            "type": "object",
            "properties": {
                "artists": {"type": "array", "items": {"type": "string"}},
                "tracks": {"type": "array", "items": {"type": "string"}},
                "albums": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "default": 40},
            },
        },
    },
    {
        "name": "play",
        "description": (
            "Hand back ONE named track for immediate playback. Use when they "
            "ask to hear something specific. Returns that track alone — not "
            "things that sound like it, and no wildcard."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "track title, or its id"}},
            "required": ["query"],
        },
    },
    {
        "name": "library_overview",
        "description": (
            "Size of the library, number of distinct artists, who is most "
            "represented, and the available axes. For orientation questions."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "refine",
        "description": (
            "Adjust the current search from the user's feedback. Moves the query "
            "toward tracks they kept, away from ones they rejected, and along "
            "named attribute axes. This is how you express 'more X' and "
            "especially 'less X' — never phrase a reduction as a description."),
        "input_schema": {
            "type": "object",
            "properties": {
                "adjust": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "axis": {"type": "string"},
                            "sign": {"type": "string", "enum": ["+", "-"]},
                            "magnitude": {"type": "string",
                                          "enum": ["slight", "moderate", "strong"]},
                        },
                        "required": ["axis", "sign"],
                    }},
                "keep_track_ids": {"type": "array", "items": {"type": "string"}},
                "reject_track_ids": {"type": "array", "items": {"type": "string"}},
                "descriptions": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 3,
                    "description": "only for critique that fits no axis"},
                "k": {"type": "integer", "default": 3},
            },
        },
    },
]


def _fmt(results):
    out = []
    for r in results:
        d = dict(track_id=r["track_id"], name=r["name"], artist=r["artist"])
        if r.get("profile"):
            d["profile"] = r["profile"]
        if r.get("wildcard"):
            d["wildcard"] = True
        out.append(d)
    return out


class CrateChat(object):
    """Holds one Crate, one Session and the message history."""

    def __init__(self, crate_model=None, model=None):
        self.crate = crate_model or crate.Crate()
        self.session = self.crate.new_session()
        self.model = model if model is not None else MODEL
        self.messages = []
        self.last_results = []      # structured results from the most recent tool call
        self._searched = False      # has any search happened in this conversation
        # Token accounting for the whole conversation. cache_read is the number
        # that tells you prompt caching is live; if it stays 0 the breakpoints
        # are not matching and you are paying full price every call.
        self.usage = {"calls": 0, "input": 0, "output": 0,
                      "cache_write": 0, "cache_read": 0}
        # What the model did this turn, in order, for the stage to animate:
        # the descriptions it wrote, the artists it resolved, the axes it moved.
        # Already computed internally; this just keeps it instead of discarding it.
        self.steps = []
        self._client = None

    def system_prompt(self):
        return SYSTEM.format(n=self.crate.n, axes=", ".join(self.crate.axis_names()),
                             axis_trust=self._axis_trust())

    def _axis_trust(self):
        """How much weight each axis deserves, taken from the validation report
        rather than hardcoded, so it cannot drift away from what was measured."""
        rep = self.crate.axis_report or {}
        good = sorted(a for a in self.crate.axes
                      if rep.get(a, {}).get("verdict") == "verified")
        soft = sorted(a for a in self.crate.axes
                      if rep.get(a, {}).get("verdict") == "unverified")
        lines = []
        if good:
            lines.append("Axes checked against features measured from the actual "
                         "waveforms: %s. These are dependable for relative "
                         "placement." % ", ".join(good))
        if soft:
            lines.append("Axes with no acoustic proxy, judged by their extremes "
                         "only: %s. Treat as a weak hint; let what you know about "
                         "the recording lead." % ", ".join(soft))
        return "\n".join(lines) if lines else ""

    # -- tool dispatch -------------------------------------------------------
    def call_tool(self, name, args):
        """Run one tool and return the JSON-able payload Claude gets back."""
        s, c = self.session, self.crate
        k = int(args.get("k") or 3)
        try:
            out = self._call_tool(name, args, s, c, k)
            if out.get("results"):
                self.last_results = c.last_results
            self.steps.append(self._step(name, args, out))
            return out
        except Exception as e:
            return dict(error="%s: %s" % (type(e).__name__, e))

    def _call_tool(self, name, args, s, c, k):
        try:
            if name == "search_by_description":
                res = c.search_by_description(s, args["descriptions"], k=k,
                                              wildcard=args.get("wildcard", True))
                return dict(results=_fmt(res), fit=c.last_fit)

            if name == "search_by_artist":
                res, info = c.search_by_artist(s, args["artist"], k=k)
                return dict(results=_fmt(res), seeded_from=info.get("source"),
                            seed_tracks=[x["name"] for x in info.get("seeds", [])],
                            n_in_library=info.get("n_in_library"),
                            fit=(c.last_fit if res else None),
                            error=info.get("error"))

            if name == "search_by_track":
                res, info = c.search_by_track(s, args["query"], k=k)
                return dict(results=_fmt(res), resolved=info.get("resolved"),
                            fit=(c.last_fit if res else None),
                            error=info.get("error"))

            if name == "add_artists":
                res, info = c.add_artists(s, args["artists"], k=k,
                                          weight=args.get("weight", "moderate"))
                return dict(results=_fmt(res), artists=[
                    dict(name=i.get("resolved"), source=i.get("source"),
                         n_in_library=i.get("n_in_library"),
                         seed_tracks=[x["name"] for x in i.get("seeds", [])],
                         error=i.get("error")) for i in info.get("artists", [])],
                    blended_into_existing_query=info.get("blended_into_existing_query"),
                    weight=info.get("weight"), error=info.get("error"))

            if name == "library_lookup":
                hits, total, per = c.lookup(args.get("artists"), args.get("tracks"),
                                            args.get("albums"),
                                            int(args.get("limit") or 40))
                # The UI should be able to play what a catalogue question
                # turned up, so these count as results even though no acoustic
                # search ran.
                self.last_results = hits
                return dict(found=total, showing=len(hits),
                            per_query=per,
                            absent=sorted(k for k, v in per.items() if not v),
                            tracks=[dict(track_id=h["track_id"], name=h["name"],
                                         artist=h["artist"], album=h["album"],
                                         matched=h["matched"]) for h in hits])

            if name == "play":
                t, n = c.play(args["query"])
                if not t:
                    return dict(error="nothing in the library matches %r"
                                      % args.get("query"))
                self.last_results = [t]
                return dict(playing=dict(track_id=t["track_id"], name=t["name"],
                                         artist=t["artist"], album=t["album"],
                                         profile=t["profile"]),
                            other_matches=max(0, n - 1))

            if name == "library_overview":
                return c.overview()

            if name == "refine":
                res, applied = c.refine(
                    s, adjust=args.get("adjust"),
                    keep=args.get("keep_track_ids"),
                    reject=args.get("reject_track_ids"),
                    descriptions=args.get("descriptions"), k=k)
                return dict(results=_fmt(res), applied=applied, fit=c.last_fit)

            return dict(error="unknown tool %r" % name)
        except Exception as e:
            return dict(error="%s: %s" % (type(e).__name__, e))

    # -- the loop ------------------------------------------------------------
    def _backend(self):
        if self._client is None:
            self._client = llm.get_backend(model=self.model or None)
        return self._client

    def say(self, user_text, verbose=True):
        """One user turn: run the tool loop until the model stops calling tools.

        Nothing is forced. An earlier version required a tool call on the
        opening turn, which did get tracks on screen quickly but made it
        impossible to answer "what Paul Simon have I got" without an acoustic
        search, or to ask a clarifying question when the request genuinely
        needed one. The prompt now steers toward acting rather than
        interviewing, and the choice is the model's.
        """
        backend = self._backend()
        self.steps = []
        self.messages.append(llm.user(user_text))
        for attempt in range(MAX_TURNS):
            reply = backend.complete(self.system_prompt(), self.messages, TOOLS)
            self._account(reply)
            self.messages.append(llm.assistant(reply))

            if not reply.tool_calls:
                return reply.text or "(no reply)"

            results = []
            for c in reply.tool_calls:
                if verbose:
                    print("   [%s %s]" % (c.name, json.dumps(c.args)[:150]))
                if c.bad_args:
                    payload = {"error": c.bad_args,
                               "hint": "re-issue the call with valid JSON arguments"}
                else:
                    payload = self.call_tool(c.name, c.args)
                    self._searched = True
                results.append(llm.tool_result(c, payload))
            self.messages.extend(results)

        return "(stopped: too many tool turns)"

    @staticmethod
    def _step(name, args, out):
        """One beat of the mechanism, in the vocabulary the animation needs."""
        st = {"tool": name, "results": [r["track_id"] for r in out.get("results", [])]}
        if name == "search_by_description":
            st["kind"] = "structuring"
            st["descriptions"] = args.get("descriptions") or []
        elif name in ("add_artists", "search_by_artist"):
            st["kind"] = "reference"
            st["artists"] = ([a.get("name") for a in out.get("artists", [])]
                             or [args.get("artist")])
            st["seeded_from"] = out.get("seeded_from") or (
                (out.get("artists") or [{}])[0].get("source"))
            st["seed_tracks"] = [t for a in (out.get("artists") or [])
                                 for t in (a.get("seed_tracks") or [])]
        elif name == "search_by_track":
            st["kind"] = "reference"
            st["resolved"] = (out.get("resolved") or {}).get("name")
        elif name == "refine":
            st["kind"] = "correction"
            ap = out.get("applied") or {}
            st["axes"] = ap.get("axes") or []
            st["kept"] = ap.get("kept", 0)
            st["rejected"] = ap.get("rejected", 0)
            st["drift_capped"] = bool(ap.get("drift_capped"))
        if out.get("fit"):
            st["fit"] = out["fit"].get("verdict")
        if out.get("error"):
            st["error"] = out["error"]
        return st

    def _account(self, reply):
        self.usage["calls"] += 1
        for k in ("input", "output", "cache_write", "cache_read"):
            self.usage[k] += int(reply.usage.get(k, 0) or 0)

    def cache_summary(self):
        """Human-readable cache effectiveness for this conversation."""
        u = self.usage
        billed = u["input"] + u["cache_write"] + u["cache_read"]
        if not billed:
            return "no usage reported yet"
        hit = 100.0 * u["cache_read"] / float(billed)
        return ("%d calls | %d uncached in, %d cache write, %d cache read "
                "(%.0f%% of input served from cache) | %d out"
                % (u["calls"], u["input"], u["cache_write"], u["cache_read"],
                   hit, u["output"]))

    def say_stream(self, user_text, verbose=False):
        """The same loop as say(), but yielding each beat as it happens.

        The stage animates from these events in real time rather than
        re-enacting a finished turn, which matters because the model takes
        seconds: those seconds become the performance instead of a spinner.

        Yields dicts with a "type": structuring | reference | correction |
        tracks | reply | error | done.
        """
        backend = self._backend()
        self.steps = []
        self.messages.append(llm.user(user_text))

        for attempt in range(MAX_TURNS):
            try:
                reply = backend.complete(self.system_prompt(), self.messages, TOOLS)
            except Exception as e:
                yield {"type": "error", "message": "%s: %s" % (type(e).__name__, e)}
                return
            self._account(reply)
            self.messages.append(llm.assistant(reply))

            if not reply.tool_calls:
                yield {"type": "reply", "text": reply.text or ""}
                yield {"type": "done", "usage": self.usage}
                return

            results = []
            for c in reply.tool_calls:
                if c.bad_args:
                    payload = {"error": c.bad_args,
                               "hint": "re-issue the call with valid JSON arguments"}
                    yield {"type": "error", "message": "malformed tool arguments",
                           "recoverable": True}
                else:
                    # Announce the intent before doing the work, so the stage can
                    # show the structuring while the search is still running.
                    if c.name == "search_by_description":
                        ds = c.args.get("descriptions") or []
                        # The literal strings handed to CLAP's text encoder,
                        # template included. Shown verbatim on the stage: the
                        # whole point of the explainer is that you see the real
                        # input, not a paraphrase of it.
                        yield {"type": "structuring", "descriptions": ds,
                               "embedded": [taste_axes.TEMPLATE % d for d in ds],
                               "template": taste_axes.TEMPLATE,
                               "dims": 512, "averaged": len(ds) > 1}
                    elif c.name in ("add_artists", "search_by_artist"):
                        yield {"type": "reference",
                               "artists": (c.args.get("artists")
                                           or [c.args.get("artist")])}
                    elif c.name == "search_by_track":
                        yield {"type": "reference", "track": c.args.get("query")}
                    elif c.name == "refine":
                        yield {"type": "correction",
                               "axes": c.args.get("adjust") or []}
                    payload = self.call_tool(c.name, c.args)
                    self._searched = True
                    yield {"type": "tracks", "results": self.last_results,
                           "step": self.steps[-1] if self.steps else None}
                results.append(llm.tool_result(c, payload))
            self.messages.extend(results)

        yield {"type": "reply", "text": "(stopped: too many tool turns)"}
        yield {"type": "done", "usage": self.usage}

    def trail(self):
        """The dialogue as points on the taste map, for the frontend."""
        return self.session.trail


def _ellipsis(t, n):
    t = str(t)
    return t if len(t) <= n else t[:n - 1].rstrip() + "\u2026"


def _table(results, indent="      "):
    """Fixed-width track table that truncates on a word boundary with an ellipsis
    rather than slicing mid-word, which read as broken output."""
    if not results:
        return [indent + "(nothing returned)"]
    rows, n = [], 0
    for r in results:
        if r.get("wildcard"):
            rows.append("%s*  %-34s  %s   <- wildcard"
                        % (indent, _ellipsis(r["name"], 34),
                           _ellipsis(r["artist"], 30)))
        else:
            n += 1
            rows.append("%s%d. %-34s  %s" % (indent, n, _ellipsis(r["name"], 34),
                                             _ellipsis(r["artist"], 30)))
    return rows


DEMO = [
    ("the opening request",
     "search_by_description",
     {"descriptions": ["warm mellow acoustic guitar with soft vocals and a slow tempo",
                       "gentle unhurried soul with brushed drums and warm bass"], "k": 3}),
    ("they name two artists -> blended in as a weight, not a reset",
     "add_artists", {"artists": ["Alfa Mist", "Tom Misch"], "weight": "moderate", "k": 3}),
    ("one of those artists is NOT in the library -> Apple Music reference",
     "add_artists", {"artists": ["Fela Kuti"], "weight": "moderate", "k": 3}),
    ("'groovier, and less singing' -> axis arithmetic, no prose negation",
     "refine", {"adjust": [{"axis": "groove", "sign": "+", "magnitude": "strong"},
                           {"axis": "vocal", "sign": "-", "magnitude": "moderate"}],
                "k": 3}),
    ("'the first one worked, the second did not' -> logged as feedback",
     "refine", {"adjust": [{"axis": "warmth", "sign": "+", "magnitude": "slight"}],
                "k": 3}),
    ("an artist with nothing like it in the library",
     "add_artists", {"artists": ["Burial"], "weight": "strong", "k": 3}),
]


def demo():
    """Exercise the tool layer end to end with no API key, printing what Claude
    would see. This is the failure-case harness as much as the happy path."""
    cc = CrateChat()
    dropped = sorted(n for n, r in cc.crate.axis_report.items()
                     if r.get("verdict") == "failed")
    print("library   %d tracks" % cc.crate.n)
    print("axes      %s" % ", ".join(cc.crate.axis_names()))
    print("dropped   %s" % (", ".join(dropped) or "none"))
    print("adapter   %s" % ("loaded" if cc.crate._adapter is not None
                            else "none yet (needs logged feedback)"))
    print("session   %s\n" % cc.session.id)

    prev = None
    for caption, tool, args in DEMO:
        print("\u2500\u2500 %s" % caption)
        # feed the previous turn's top hit back as real feedback where the
        # scripted step is a refinement, so the store sees keep/reject traffic
        if tool == "refine" and prev:
            args = dict(args)
            args["keep_track_ids"] = [prev[0]["track_id"]]
            # verdict on the wildcard — the judgement worth having
            wc = [r for r in prev if r.get("wildcard")]
            if wc:
                args["reject_track_ids"] = [wc[0]["track_id"]]
            elif len(prev) > 1:
                args["reject_track_ids"] = [prev[1]["track_id"]]
        out = cc.call_tool(tool, args)
        if out.get("error"):
            print("   ! %s" % out["error"])
        for line in _table(out.get("results", [])):
            print(line)
        for a in out.get("artists", []) or []:
            if a.get("error"):
                print("      ref  %-18s unresolved: %s" % (a.get("name"), a["error"]))
            else:
                print("      ref  %-18s %-13s %s"
                      % (_ellipsis(a.get("name"), 18), a.get("source"),
                         ", ".join(_ellipsis(t, 24) for t in a.get("seed_tracks", []))))
        if out.get("blended_into_existing_query") is not None:
            print("      blended into the running query: %s (weight %s)"
                  % (out["blended_into_existing_query"], out.get("weight")))
        if out.get("applied"):
            ap = out["applied"]
            bits = ["%s%s" % (x["sign"], x["axis"]) for x in ap["axes"]]
            print("      applied  %s | kept %d, rejected %d%s%s"
                  % (" ".join(bits) or "(feedback only)", ap["kept"], ap["rejected"],
                     "  [drift capped]" if ap.get("drift_capped") else "",
                     "  [unknown: %s]" % ",".join(ap["unknown_axes"])
                     if ap["unknown_axes"] else ""))
        if out.get("fit"):
            print("      fit      %s" % out["fit"]["verdict"])
        prev = out.get("results") or prev
        print()

    if cc.trail():
        print("trail (what the map draws)")
        for i, pt in enumerate(cc.trail(), 1):
            print("   %d. (%7.2f, %7.2f)  %s" % (i, pt["x"], pt["y"],
                                                 _ellipsis(pt["label"], 60)))
    try:
        import crate_store
        print()
        crate_store.stats()
    except Exception:
        pass


def main():
    if "--demo" in sys.argv:
        return demo()
    cc = CrateChat()
    print("library: %d tracks | axes: %s"
          % (cc.crate.n, ", ".join(cc.crate.axis_names())))
    if cc.crate.axis_report:
        dropped = [n for n, r in cc.crate.axis_report.items()
                   if r.get("verdict") == "failed"]
        if dropped:
            print("dropped by validation: %s" % ", ".join(sorted(dropped)))
    print("type a request, or 'trail' to see the path, Ctrl-D to quit\n")
    while True:
        try:
            line = raw_input("you > ") if sys.version_info[0] == 2 else input("you > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line.strip():
            continue
        if line.strip() == "trail":
            for p in cc.trail():
                print("   (%.2f, %.2f)  %s" % (p["x"], p["y"], p["label"]))
            continue
        print("\n" + cc.say(line) + "\n")


if __name__ == "__main__":
    main()
