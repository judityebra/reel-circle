# Reel Circle

A group movie recommender: a group of people describe what they want to watch tonight, and Reel Circle turns their histories and requests into a fair, explainable shortlist.

## People

**Person**:
One member of a group, with their own ratings and watch history.
_Avoid_: Profile (except as the code type), viewer, member

**Group**:
The people choosing a movie together for one viewing.
_Avoid_: Circle (the product name only), party, team

**Room**:
A shareable snapshot link of a group's shortlist, without anyone's ratings or watch history.
_Avoid_: Session, lobby

**User**:
Whoever is typing in the conversation. Not necessarily every person in the group.
_Avoid_: Using "user" for group members

## Agents

**Recommendation agent**:
The server-side agent that turns a conversation turn into a validated shortlist from the candidate pool.
_Avoid_: "The agent" on its own, server agent

**Lab agent**:
The in-browser agent that trains taste models on a group's ratings, asks taste questions and proposes a final pick.
_Avoid_: Decision agent, "the agent" on its own

## Requests

**Conversation**:
The sequence of turns one user has with the recommendation agent about one viewing.
_Avoid_: Chat, session, thread (implementation term)

**Hard constraint**:
A condition every recommended movie must meet: maximum runtime, excluded genres, year range, languages, requested genres (any of) and platforms (any of).
_Avoid_: Filter, rule

**Preference**:
A soft signal that changes ranking but never removes a movie, such as a mood.
_Avoid_: Constraint, filter

**Candidate pool**:
The movies the browser offers for one turn, with everything anyone in the group has watched already removed.
_Avoid_: Catalog (the full source), candidates list

**Relaxation**:
A proposal to loosen exactly one hard constraint when no movie in the candidate pool meets them all. Excluded genres are never relaxed.
_Avoid_: Fallback, broadening

**Clarifying question**:
A question the recommendation agent asks when a turn gives it nothing usable to act on.
_Avoid_: Follow-up, prompt

**Pause**:
The recommendation agent waiting on the user, either for a clarifying question or for a relaxation decision. Any new message from the user ends it.
_Avoid_: Interrupt (implementation term), approval gate (the lab agent's term)

## Evaluation

**Scenario**:
A scripted multi-turn conversation with the expected outcome of each turn.
_Avoid_: Test case, example

**Successful turn**:
A turn with no hard-constraint violations, every earlier exclusion still in force, pauses exactly where the scenario expects, at least one movie or a relaxation or an honest "nothing fits", and within the step limit. A scenario passes only when every turn succeeds.
_Avoid_: Correct answer, pass (without saying turn or scenario)
