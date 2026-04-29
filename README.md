# Traveling Salesdog

Minimal Julia + JuMP optimizer for a dog walking schedule.

The model chooses one activity for each morning and night walk over seven days. It
rewards variety and run/play opportunities while penalizing dog effort: total
mileage under foot.

## Setup

Install Julia, then from this directory run:

```sh
julia +1.11 --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run

```sh
julia +1.11 --project=. src/schedule.jl
```

Use a different CSV file:

```sh
julia +1.11 --project=. src/schedule.jl --activities path/to/activities.csv
```

Tune constraints and weights:

```sh
julia +1.11 --project=. src/schedule.jl \
  --max-daily-distance 2.0 \
  --w-area 4 \
  --w-run 1.5 \
  --w-distance 2
```

## CSV Format

The activity CSV must contain these columns. `distance_miles` is the one-way
distance to the activity; the model counts `2 * distance_miles` for total mileage
under foot.

```csv
activity_id,distance_miles,area,has_run_area
R1,1.1,park,1
```

`has_run_area` must be `0` or `1`.

## Model

Decision variables:

- `x[d, w, a]`: activity `a` is chosen on day `d` for walk slot `w`
- `y[a]`: area `a` is visited at least once during the week

Constraints:

- exactly one activity for each morning and night walk
- combined daily round-trip distance must be at or below `--max-daily-distance`
- at least one run/play activity per day
- no same activity for both walks on the same day
- novelty rule: an activity used yesterday cannot be used today
- area usage is linked to selected activities

Objective:

```text
maximize
  w_area * unique areas
+ w_run * run/play walks
- w_distance * total distance
```

The model assumes `distance_miles` is the one-way pavement exposure needed to get
to the activity. It doubles that distance for the daily cap, objective penalty,
and summary output. Novelty is modeled as a one-day activity cooldown.
