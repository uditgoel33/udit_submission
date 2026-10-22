# Instructions

Hello, and thanks for taking the time to interview with us at Tracksuit! This is a take-home exercise we'd like you to complete. This is an open-ended problem that is designed to showcase your thoughtfulness and creativity. To avoid spending too much time on this task, we encourage you to timebox your work to a few hours.

<!-- deno-fmt-ignore-start -->
> [!Note]
> This task is meant to give us something to talk over in your technical
> interview. It is a challenging but realistic example of the kind of
> problems you may tackle at Tracksuit and you're encouraged to use AI
> to assist you. Finally, remember: Nothing is designed to trick you.
> If you have any questions, please reach out!
<!-- deno-fmt-ignore-end -->

## Background

To maintain its trajectory of world-class growth, Tracksuit needs to continually
innovate to speed up our sales process. One of the biggest current bottlenecks 
to sales efficiency is **incidence testing**.

Tracksuit runs category-based surveys for consumer brands. This means that we 
guarantee a minimum of 200 respondents/month in the customer's category of 
choice.
However, not all categories are equal. It's much easier to find 200 people
who ate fast food in the last month than 200 people who bought electric 
tricycles. 
In order to protect our costs and ensure we have enough respondents to service
the customer, we only sign-on customers with an incidence rate over a set percentage.
Incidence tests are surveys we run to check the approximate incidence rate
of a category.

> #### 💡 What is an incidence rate?
> Each Tracksuit category is defined by a **category qualifier**.  
> For fast food, this might be: *"Did you purchase fast food in the last 3 months?"*
>
> The **incidence rate** is the percentage of people who respond positively — either:
> - **"Yes"**, or  
> - **"No, but I am open or likely to in the near future."**
>
> For example, if we asked the category qualifier to 220 people and 20 responded  
> "Yes" or "No, but... near future", the category would have an **incidence rate of 10%**.  
> In this case, we would **not** be able to sell the category.

Incidence tests are both expensive and slow. They're expensive because we have
to survey so many people - even before we sign on the customer. And they're slow 
because, since they're expensive, we want to do them less frequently. Currently, 
it can take up to a week to get results from incidence tests. Ideally, we could
get incidence test results for Sales live and on demand.

In this task, imagine you're a Data Scientist at Tracksuit building the first version of this capability.

## The Task

Build a method that, given a category qualifier, predicts the incidence rate that a real incidence test would return, together with an honest account of how uncertain that prediction is.

We explicitly have no preferred model family. You may use algorithms from statistics, machine learning, or any other field. We care considerably more about the performance and argument you make for your approach than about your specific model.

There are three deliverables.

### 1. Predictions

`prediction_set.csv` contains 110 category-geographies that have been held out of `incidence_dataset.csv`. 

Produce a `predictions.csv` with one row per `prediction_id` and these columns:

| column | meaning |
| --- | --- |
| `prediction_id` | as given in `prediction_set.csv` |
| `point_estimate` | your best estimate of the incidence rate, between 0 and 1 |
| `lower_80` | lower bound of an 80% prediction interval |
| `upper_80` | upper bound of an 80% prediction interval |

See `predictions_example.csv` for the format. We will score these, so please make sure the file parses and covers every row.

The interval is not optional, and we will check whether it is *calibrated* - that is, whether roughly 80% of the true values actually fall inside the intervals you give us. An interval that is too wide is as much of a problem as one that is too narrow.

### 2. Method and validation

A short write-up (a notebook, a markdown file, whatever suits) covering how your method works and, more importantly, **how you convinced yourself it works**.

We have deliberately not given you a training/validation split. Deciding how to construct one from `incidence_dataset.csv`, and justifying that decision, is part of what we're assessing. Tell us what error you expect on the prediction set before you see the results, and why.

Please also tell us what your interval represents. Different sources of uncertainty behave differently as we collect more data, and we're interested in which ones you think you're capturing.

Finally, note limitations with the data, results, or your approach that you'd prioritise if you had further time.

### 3. Application

**Air Conditioners** is tracked in Australia but a prospect wants to track it in the **United States**. What incidence rate would you predict, and how much would you trust it?

---

Since we'll be reviewing your work asynchronously, please follow the instructions in the [README](./README.md) to submit your task. You're welcome to use your programming language and framework of choice but please ensure your full solution is reproducible from your code.

We recognise that this task is ambitious. You're actively encouraged to use any AI tools (e.g. Codex, Claude Code) to help you with this take home task, similarly to how you would as an employee at Tracksuit. However, like any code you write at Tracksuit, you are responsible for the quality of your code and your results. Make sure you can interpret and explain any code, assumptions, and results that you submit. Please note that we will be comparing your results to a naive Claude Code solution based on this repository. You'll need to contribute additional insight or intuition to achieve a strong performance.

In addition, if your approach would benefit from access to an LLM, please reach out and we'll share a one-off API key with you for your assignment.

## The Data

`incidence_dataset.csv` contains 833 category-geographies from our live tracking, as at the September 2024 wave. Each row is one category asked in one geography.

A few things worth knowing about how this data is produced:

- Every row is a **three month roll**. For `wave_date = 2024-09-01`, the metrics aggregate responses collected over July, August and September 2024. Every row in this dataset shares that same wave date.
- `percentage_3` is **an estimate, not a measurement**. It is the result of asking a finite sample of respondents, and the size of that sample varies a great deal across rows - from roughly 220 respondents to over 16,000.
- Weights are approximately the number of respondents, adjusted for discrepancies between the sample distribution and the national population according to their census. We weight using the random iterative method (RIM).
- The geographies are not evenly represented. Of the 833 rows, Australia accounts for 329, the United States 204, New Zealand 200, the United Kingdom 88, and Canada 12.
- Qualifier wording is not standardised across categories. Recall windows, note text, and the set of answer options that count as qualifying all vary, and they were chosen by a human at the time the category was set up.

**_Data dictionary: `incidence_dataset.csv`_**
- `category_name`: the name of the category
- `geography_name`: the geography the category was asked in
- `wave_date`: the month the data represents (constant across this dataset)
- `question_text`: the category qualifier that was asked to survey respondents
- `qualifying_options`: all answer options that result in the respondent qualifying for the category, separated by `|`
- `disqualifying_options`: all answer options that result in the respondent being disqualified, separated by `|`
- `weight_3`: the weight of respondents who qualified for the category (numerator)
- `base_weight_3`: the weight of all respondents who were asked the category qualifier (denominator)
- `percentage_3`: the incidence rate, `weight_3 / base_weight_3`. Note that some rows carry this rounded to two decimal places, so the division will not always reproduce it exactly.

**_Data dictionary: `prediction_set.csv`_**
- `prediction_id`: a stable identifier for the row, to be echoed back in your `predictions.csv`
- all of `category_name`, `geography_name`, `wave_date`, `question_text`, `qualifying_options`, `disqualifying_options`, as above

The three weight and percentage columns are withheld for this file. No category in `prediction_set.csv` appears in `incidence_dataset.csv`.

We hope you enjoy this take home task. If you have any questions - please reach out! We look forward to reviewing your submission.
