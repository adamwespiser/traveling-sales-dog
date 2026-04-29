using CSV
using DataFrames
using HiGHS
using JuMP

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
const WALK_TIMES = ["morning", "night"]

function parse_args(args)
    options = Dict(
        "activities" => "data/activities.csv",
        "max-daily-distance" => "2.0",
        "w-area" => "4.0",
        "w-run" => "1.5",
        "w-distance" => "2.0",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if !startswith(arg, "--")
            error("Unexpected argument: $arg")
        end

        key_value = split(arg[3:end], "=", limit = 2)
        if length(key_value) == 2
            options[key_value[1]] = key_value[2]
        else
            i += 1
            i > length(args) && error("Missing value for argument: $arg")
            options[key_value[1]] = args[i]
        end
        i += 1
    end

    return options
end

function load_activities(path)
    activities = CSV.read(path, DataFrame)
    required = [:activity_id, :distance_miles, :area, :has_run_area]
    missing = setdiff(required, Symbol.(names(activities)))
    isempty(missing) || error("Missing required CSV columns: $(join(missing, ", "))")

    activities.activity_id = string.(activities.activity_id)
    activities.area = string.(activities.area)
    activities.has_run_area = Int.(activities.has_run_area)
    activities.distance_miles = Float64.(activities.distance_miles)

    any(activities.distance_miles .< 0) && error("distance_miles must be non-negative")
    any((activities.has_run_area .!= 0) .& (activities.has_run_area .!= 1)) && error("has_run_area must be 0 or 1")
    length(unique(activities.activity_id)) == nrow(activities) || error("activity_id values must be unique")

    return activities
end

function solve_schedule(activities; max_daily_distance, weights)
    day_count = length(DAYS)
    walk_count = length(WALK_TIMES)
    activity_count = nrow(activities)
    activity_indexes = 1:activity_count
    day_indexes = 1:day_count
    walk_indexes = 1:walk_count
    areas = sort(unique(activities.area))

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[day_indexes, walk_indexes, activity_indexes], Bin)
    @variable(model, y[areas], Bin)
    round_trip_distance = 2 .* activities.distance_miles

    # Pick exactly one activity for each walk slot: morning and night.
    @constraint(model, [d in day_indexes, w in walk_indexes], sum(x[d, w, a] for a in activity_indexes) == 1)

    # Keep the combined morning plus night pavement distance within the daily limit.
    @constraint(model, [d in day_indexes],
        sum(round_trip_distance[a] * x[d, w, a] for w in walk_indexes, a in activity_indexes) <= max_daily_distance
    )

    # Require at least one daily activity with a run/play area.
    @constraint(model, [d in day_indexes],
        sum(activities.has_run_area[a] * x[d, w, a] for w in walk_indexes, a in activity_indexes) >= 1
    )

    # Prevent using the same activity in both walk slots on the same day.
    @constraint(model, [d in day_indexes, a in activity_indexes], sum(x[d, w, a] for w in walk_indexes) <= 1)

    # Novelty rule: if an activity was used at any time yesterday, do not use it today.
    @constraint(model, [d in 2:day_count, a in activity_indexes],
        sum(x[d, w, a] for w in walk_indexes) + sum(x[d - 1, w, a] for w in walk_indexes) <= 1
    )

    for area in areas
        area_activities = findall(==(area), activities.area)

        # If y[area] is 1, at least one selected activity must visit that area.
        @constraint(model, y[area] <= sum(x[d, w, a] for d in day_indexes, w in walk_indexes, a in area_activities))

        # If any activity in this area is selected, mark this area as visited.
        @constraint(model, [d in day_indexes, w in walk_indexes, a in area_activities], y[area] >= x[d, w, a])
    end

    @objective(model, Max,
        weights.area * sum(y[area] for area in areas) +
        weights.run * sum(activities.has_run_area[a] * x[d, w, a] for d in day_indexes, w in walk_indexes, a in activity_indexes) -
        weights.distance * sum(round_trip_distance[a] * x[d, w, a] for d in day_indexes, w in walk_indexes, a in activity_indexes)
    )

    optimize!(model)

    if !is_solved_and_feasible(model)
        error("Optimization failed with status: $(termination_status(model))")
    end

    chosen = DataFrame(
        day = String[],
        walk_time = String[],
        activity_id = String[],
        area = String[],
        one_way_distance_miles = Float64[],
        round_trip_distance_miles = Float64[],
        has_run_area = Int[],
    )

    for d in day_indexes
        for w in walk_indexes
            a = only(filter(a -> value(x[d, w, a]) > 0.5, activity_indexes))
            push!(chosen, (
                DAYS[d],
                WALK_TIMES[w],
                activities.activity_id[a],
                activities.area[a],
                activities.distance_miles[a],
                round_trip_distance[a],
                activities.has_run_area[a],
            ))
        end
    end

    optimal_solution_count = count_optimal_solutions(activities; max_daily_distance, weights)
    raw_search_space_size = BigInt(activity_count)^(day_count * walk_count)

    return chosen, objective_value(model), optimal_solution_count, raw_search_space_size
end

function count_optimal_solutions(activities; max_daily_distance, weights)
    scale = 1_000_000
    activity_count = nrow(activities)
    activity_indexes = 1:activity_count
    areas = sort(unique(activities.area))
    area_bit = Dict(area => UInt64(1) << (i - 1) for (i, area) in enumerate(areas))
    round_trip_distance = 2 .* activities.distance_miles

    pairs = []
    for morning in activity_indexes, night in activity_indexes
        morning == night && continue
        distance = round_trip_distance[morning] + round_trip_distance[night]
        distance <= max_daily_distance || continue
        run_count = activities.has_run_area[morning] + activities.has_run_area[night]
        run_count >= 1 || continue

        activity_mask = (UInt64(1) << (morning - 1)) | (UInt64(1) << (night - 1))
        area_mask = area_bit[activities.area[morning]] | area_bit[activities.area[night]]
        score = weights.run * run_count - weights.distance * distance

        push!(pairs, (
            activity_mask = activity_mask,
            area_mask = area_mask,
            score = round(Int, score * scale),
        ))
    end

    # State is (yesterday's activity mask, areas visited so far, score excluding area reward).
    states = Dict{Tuple{UInt64, UInt64, Int}, BigInt}((UInt64(0), UInt64(0), 0) => BigInt(1))

    for _ in DAYS
        next_states = Dict{Tuple{UInt64, UInt64, Int}, BigInt}()
        for ((previous_activity_mask, visited_area_mask, score), count) in states
            for pair in pairs
                (previous_activity_mask & pair.activity_mask) == 0 || continue

                key = (
                    pair.activity_mask,
                    visited_area_mask | pair.area_mask,
                    score + pair.score,
                )
                next_states[key] = get(next_states, key, BigInt(0)) + count
            end
        end
        states = next_states
    end

    best_score = typemin(Int)
    best_count = BigInt(0)
    for ((_, visited_area_mask, score), count) in states
        area_score = round(Int, weights.area * count_ones(visited_area_mask) * scale)
        total_score = score + area_score

        if total_score > best_score
            best_score = total_score
            best_count = count
        elseif total_score == best_score
            best_count += count
        end
    end

    return best_count
end

function print_schedule(schedule, objective, optimal_solution_count, raw_search_space_size)
    println("Weekly dog-walking schedule")
    println("===========================")
    for day in DAYS
        println(day * ":")
        for row in eachrow(filter(:day => ==(day), schedule))
            run_label = row.has_run_area == 1 ? "run/play" : "sniff walk"
            println("  ", rpad(row.walk_time * ":", 9), "$(row.activity_id) ($(row.area), $(run_label), $(round(row.round_trip_distance_miles, digits = 2)) mi under foot)")
        end
    end

    total_distance = sum(schedule.round_trip_distance_miles)
    run_days = sum(schedule.has_run_area)
    unique_areas = length(unique(schedule.area))

    println()
    println("Summary")
    println("=======")
    println("objective value:      ", round(objective, digits = 3))
    println("mileage under foot:   ", round(total_distance, digits = 2), " miles")
    println("unique areas:         ", unique_areas)
    println("run/play walks:       ", run_days)
    println("instances at optimum: ", optimal_solution_count)
    println("raw search space:     ", raw_search_space_size, " schedules")
end

function main(args = ARGS)
    options = parse_args(args)
    activities = load_activities(options["activities"])
    weights = (
        area = parse(Float64, options["w-area"]),
        run = parse(Float64, options["w-run"]),
        distance = parse(Float64, options["w-distance"]),
    )

    schedule, objective, optimal_solution_count, raw_search_space_size = solve_schedule(
        activities;
        max_daily_distance = parse(Float64, options["max-daily-distance"]),
        weights,
    )
    print_schedule(schedule, objective, optimal_solution_count, raw_search_space_size)
end

main()
