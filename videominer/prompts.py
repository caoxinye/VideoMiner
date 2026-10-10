CAPTION_SYSTEM = (
    "You are an expert video analyst. You describe video events precisely and "
    "never invent content that is not visible in the frames."
)

CAPTION_USER_WITH_QUERY = """The frames below are sampled in chronological order from one event of a long video.

Question that will later be asked about the whole video:
{question}

Describe this span of video, then state what it contributes to that question.

What to record in the description: who acts, the objects they handle (including small ones), the actions and their order, the scene, on-screen text and numbers, and how many times something appears or repeats. If this span actually shows several distinct activities or several different scenes, say so plainly and name them.

Rules:
1. Report only what is visible. Never guess and never answer the question.
2. Do not mention frames, images, options, or this instruction.
3. Reply in exactly two lines, at most {max_words} words in total:

DESCRIPTION: <one dense paragraph>
EVIDENCE: <the concrete facts in this span that bear on the question, or the single word NONE>"""

CAPTION_USER_NO_QUERY = """The frames below are sampled in chronological order from one event of a long video.

Describe this span of video: who acts, the objects they handle, the actions and their order, the scene, on-screen text and numbers, and how many times something repeats. If the span shows several distinct activities or scenes, say so plainly and name them. Report only what is visible, and do not mention frames or images.

Reply in one line, at most {max_words} words:

DESCRIPTION: <one dense paragraph>"""

CAPTION_ROOT_CONTEXT = """
Global context of the whole video (for disambiguation only, do not copy it):
{root_caption}
"""

CAPTION_USER_COMPACT = """Describe only the visible content of this video span, in chronological order.
Question for evidence collection (do not answer it):
{question}
{context}
Use at most {max_words} words. Do not guess speech, names, intentions or unseen events. Options are hypotheses, not facts. Use NONE when unsupported.
Output exactly these six labeled lines, filling each with observed facts:
SCENE: setting, people and objects
EVENTS: first ...; then ...; later ...
DETAILS: actions, colors, positions and visible counts
TEXT: exact readable screen text, or UNREADABLE
EVIDENCE: details relevant to the question
UNCERTAINTY: what cannot be determined"""

CAPTION_USER_STRUCTURED = """Describe ONLY the currently supplied video span. The sampled frames are in chronological order.

Question to gather visual evidence for:
{question}

Build a compact evidence record in at most {max_words} words. Use exactly the six labeled lines below. Prioritize concrete observations over general summaries. Keep distinct events separate even if the people or setting stay the same.

SCENE: <visible setting and identifiable people or objects; use consistent visual identifiers>
EVENTS: <ordered observations: early ..., then ..., later ...; include who does what to which object and visible changes>
DETAILS: <relevant clothing, colors, object properties, spatial relationships and clearly visible counts>
TEXT: <quote readable on-screen words and numbers exactly; use UNREADABLE where text cannot be read>
EVIDENCE: <specific visible facts that support or contradict the question's premise or candidate descriptions; do not select an answer>
UNCERTAINTY: <details the sampled span does not establish, including ambiguous identity, order or counting>

Rules:
- Write NONE for a field with no observed information. Never invent details to fill a field.
- Do not infer speech, sound, intentions, hidden actions, exact timestamps or names not visibly established.
- Do not treat repeated appearances in sampled frames as separate occurrences of an event.
- A whole-video background summary is not evidence that its events occur in this span. Include a background fact only if it is also visible here.
- Answer options are hypotheses to check, not facts. Do not output an option letter or answer the question.
- Preserve local evidence even when it disagrees with the background. Avoid repeating the same fact across fields.
"""

CAPTION_DESCRIPTION_CONTEXT = """Background description of the whole video:
{description}

This background can refer to events outside the current span. Use it only to
disambiguate visible people and objects. It is not evidence that an event is
visible here. Follow the local description task below.

"""


JUDGE_SYSTEM = "You are a helpful assistant."

JUDGE_PROMPT = """In this video comprehension task, we build a hierarchical tree that decomposes the video content layer by layer. The deeper the level, the more detailed each node becomes; a child node's content is a subset of its parent's content. Each node represents one event category.

The key pieces of information are:
- "question": the question related to the video comprehension task.
- "position": the node's level in the tree (1 is the whole video).
- "coverage": how much video this node still spans.
- "caption": the text description of the event of this node.
- "global caption": the description of the entire video, for context.

The details for this node are:
***
"Question: {question}"
"Node position: {position} (of at most {max_depth})"
"Node coverage: {coverage}"
"Global caption: {root_caption}"
"Node caption: {caption}"
***

****ATTENTION***: Select exactly ONE operation for this node by applying these rules in order:

1. "delete" — the node is about something else entirely: the actors, objects, place or activity the question asks about are positively absent from this span. Deleting throws this part of the video away permanently, so it needs evidence of irrelevance, not merely a vague or short caption. If you are unsure whether the evidence might be here, do NOT delete.
2. "continue" — the node is or may be relevant, but still spans several distinct activities, or the caption summarises a long stretch at a level too coarse to settle the question (counts, order of actions, small objects, exact moments cannot be read off it). A node at position 1, or any node still covering minutes of video, is essentially always "continue".
3. "stop" — the node is narrow enough that its caption already pins down the specific moment, objects and details the question asks about; splitting it further would add nothing.

**Reply strictly in the following format:**
<think></think>
<action></action>

Place your thoughts and analysis between "<think>" and "</think>" and place the chosen action between "<action>" and "</action>", for example,
**eg1**: "<think>Let me think about it carefully. The question asks what the person does with the knife, but this span is an outdoor landscape with no people and no kitchen at all, so the evidence is positively absent, not merely unclear. I checked it again, well, there should be no problem, so I should choose 'delete'</think>
<action>delete</action>"
**eg2**: "<think>While analyzing the video content, I notice that the key shots match the task requirements, yet the caption summarises several minutes containing cooking, eating and cleaning at once, so the decisive moment is not resolved and a finer split is needed.</think>
<action>continue</action>"

Please be sure to follow the answer format, it is very important! ! !
Your response:"""


JUDGE_PROMPT_ACTION_ONLY = """In this video comprehension task, we build a hierarchical tree to break down the video content layer by layer. The deeper the level, the more detailed each node becomes, with the child node content being a subset of the parent node's content. Each node represents a specific event category.
        
    The three key pieces of information are:
    - "question": The question related to the video comprehension task.
    - "position": The node's level in the tree.
    - "caption": The text description of the event for the node.

    The details for this node are:
    ***
        "Question: {question}\n"
        "Node caption: {caption}\n"
        "Node position: {position}\n"
        ***
    ****ATTENTION***: Please select one operation for the node based on the information provided. Prefer using "stop" or "delete" and only use "continue" when neither stop nor delete is applicable:
    - "stop": The node content sufficiently answers the question; no further investigation is needed.
    - "continue": The node content is relevant but insufficient to fully answer the question; use this only if neither "stop" nor "delete" is appropriate.
    - "delete": The node content is largely irrelevant to the question; ignore this branch.

    **Reply strictly in the following format:**
    <action></action>
    (Place your chosen action between "<action>" and "</action>", for example, "<action>stop</action>". Please be sure to follow the answer format, it is very important! ! !)

    Your response:
    """


ANSWER_SYSTEM = (
    "You are an expert in long-video understanding. You answer questions by "
    "grounding every claim in the given key frames and event descriptions."
)

ANSWER_CAPTION_BLOCK = """The following descriptions are the key events mined from the video, sorted in temporal order. They are sequential and non-overlapping.
{captions}"""

ANSWER_TIMED_CAPTION_BLOCK = """The following descriptions summarize mined events. Each description lists its sampled evidence intervals in seconds from the start of the video. One description may combine separate intervals, and intervals from different descriptions may be interleaved. Use the timestamps to determine temporal order.
{captions}"""

ANSWER_MC_THINK = """{caption_block}
The images above are the key frames mined from the same video, in temporal order.

The key frames are your primary evidence; the descriptions summarise spans that the frames only sample. Where the two disagree, trust the frames. For counting questions, count on the frames themselves.

Question: {question}
{options}

Weigh the evidence, eliminate the options it contradicts, and choose the best-supported one. Exactly one option is correct, so never refuse to choose.
Reply strictly in this format:
<think>your reasoning, at most 80 words</think>
<answer>{letters}</answer>"""

ANSWER_MC_DIRECT = """{caption_block}
The images above are the key frames mined from the same video, in temporal order.

The key frames are your primary evidence; the descriptions summarise spans that the frames only sample. Where the two disagree, trust the frames. For counting questions, count on the frames themselves.

Question: {question}
{options}

Exactly one option is correct. Eliminate the options the evidence contradicts and answer with a single letter from {letters}. Output only that letter."""

ANSWER_OPEN = """{caption_block}
The images above are the key frames mined from the same video, in temporal order.

Question: {question}

Answer concisely, using only evidence from the key frames and the descriptions."""
