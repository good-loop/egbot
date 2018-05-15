
Agenda

1. Introductions.

2. Tea and biscuits

3. Project overview (Alison)

4. Some key metrics/goals to track.
I suggest:
 - No. of people the system engages with.
 - % of successful engagements (measured by user feedback).
 - % of Qs from a test dataset which the system can solve.
 - A target for paper(s) written.

5. Agree basic tools   
I suggest: GitHub for code, DropBox for data, email for comms, ??ZenDesk for task tracking.

6. June sprint plan. Draft below...

Draft June Sprint Plan

1. Data Collection
	- Collect 20+ examples of the Q&A we'd like to handle from both MathOverflow and Reddit.   
	Maybe just use browser bookmarks as our collection tool?   
	- ...and an equal number of threads we don't want to engage with.
	- Try to find RSS feeds or tags which provide a large & fresh supply of both.

2. Markup the examples with discourse metadata (type of message).
	- Everyone does some initial markup.
	- ...then we discuss and agree a schema.
	- Discuss tooling.

3. Create a walking skeleton
	- User can type in a question.
	- System will respond with an answer.
	- The AI behind this is very simple. Perhaps: put training data Q&A into ElasticSearch (ES). Then use ES's built-in similarity measure to pick the answer.
	- Stretch goal: Results: User can provide +/- feedback, and we also store accuracy metrics from training.

Meanwhile, we'll also give Irina a 20% time Good-Loopy project. TBD -- some ideas are:

 - Data visualisation for the user portal.
 - Data extraction from user data, e.g. Twitter or CRM data.
 - Machine learning optimisation for inventory buying and ad display.
