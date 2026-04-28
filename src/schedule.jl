using CSV
using DataFrames
using HiGHS
using JuMP

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
const WALK_TIMES = ["morning", "night"]

function parse_args(args)
    options = Dict(
        "routes" => "data/routes.csv",
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

function load_routes(path)
    routes = CSV.read(path, DataFrame)
    required = [:route_id, :distance_miles, :area, :has_run_area]
    missing = setdiff(required, Symbol.(names(routes)))
    isempty(missing) || error("Missing required CSV columns: $(join(missing, ", "))")

    routes.route_id = string.(routes.route_id)
    routes.area = string.(routes.area)
    routes.has_run_area = Int.(routes.has_run_area)
    routes.distance_miles = Float64.(routes.distance_miles)

    any(routes.distance_miles .< 0) && error("distance_miles must be non-negative")
    any((routes.has_run_area .!= 0) .& (routes.has_run_area .!= 1)) && error("has_run_area must be 0 or 1")
    length(unique(routes.route_id)) == nrow(routes) || error("route_id values must be unique")

    return routes
end

function solve_schedule(routes; max_daily_distance, weights)
    day_count = length(DAYS)
    walk_count = length(WALK_TIMES)
    route_count = nrow(routes)
    route_indexes = 1:route_count
    day_indexes = 1:day_count
    walk_indexes = 1:walk_count
    areas = sort(unique(routes.area))

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[day_indexes, walk_indexes, route_indexes], Bin)
    @variable(model, y[areas], Bin)
    round_trip_distance = 2 .* routes.distance_miles

    # Pick exactly one route for each walk slot: morning and night.
    @constraint(model, [d in day_indexes, w in walk_indexes], sum(x[d, w, r] for r in route_indexes) == 1)

    # Keep the combined morning plus night pavement distance within the daily limit.
    @constraint(model, [d in day_indexes],
        sum(round_trip_distance[r] * x[d, w, r] for w in walk_indexes, r in route_indexes) <= max_daily_distance
    )

    # Require at least one daily route with a run/play area.
    @constraint(model, [d in day_indexes],
        sum(routes.has_run_area[r] * x[d, w, r] for w in walk_indexes, r in route_indexes) >= 1
    )

    # Prevent using the same route in both walk slots on the same day.
    @constraint(model, [d in day_indexes, r in route_indexes], sum(x[d, w, r] for w in walk_indexes) <= 1)

    # Novelty rule: if a route was used at any time yesterday, do not use it today.
    @constraint(model, [d in 2:day_count, r in route_indexes],
        sum(x[d, w, r] for w in walk_indexes) + sum(x[d - 1, w, r] for w in walk_indexes) <= 1
    )

    for area in areas
        area_routes = findall(==(area), routes.area)

        # If y[area] is 1, at least one selected route must visit that area.
        @constraint(model, y[area] <= sum(x[d, w, r] for d in day_indexes, w in walk_indexes, r in area_routes))

        # If any route in this area is selected, mark this area as visited.
        @constraint(model, [d in day_indexes, w in walk_indexes, r in area_routes], y[area] >= x[d, w, r])
    end

    @objective(model, Max,
        weights.area * sum(y[area] for area in areas) +
        weights.run * sum(routes.has_run_area[r] * x[d, w, r] for d in day_indexes, w in walk_indexes, r in route_indexes) -
        weights.distance * sum(round_trip_distance[r] * x[d, w, r] for d in day_indexes, w in walk_indexes, r in route_indexes)
    )

    optimize!(model)

    if !is_solved_and_feasible(model)
        error("Optimization failed with status: $(termination_status(model))")
    end

    chosen = DataFrame(
        day = String[],
        walk_time = String[],
        route_id = String[],
        area = String[],
        one_way_distance_miles = Float64[],
        round_trip_distance_miles = Float64[],
        has_run_area = Int[],
    )

    for d in day_indexes
        for w in walk_indexes
            r = only(filter(r -> value(x[d, w, r]) > 0.5, route_indexes))
            push!(chosen, (
                DAYS[d],
                WALK_TIMES[w],
                routes.route_id[r],
                routes.area[r],
                routes.distance_miles[r],
                round_trip_distance[r],
                routes.has_run_area[r],
            ))
        end
    end

    optimal_solution_count = count_optimal_solutions(routes; max_daily_distance, weights)
    raw_search_space_size = BigInt(route_count)^(day_count * walk_count)

    return chosen, objective_value(model), optimal_solution_count, raw_search_space_size
end

function count_optimal_solutions(routes; max_daily_distance, weights)
    scale = 1_000_000
    route_count = nrow(routes)
    route_indexes = 1:route_count
    areas = sort(unique(routes.area))
    area_bit = Dict(area => UInt64(1) << (i - 1) for (i, area) in enumerate(areas))
    round_trip_distance = 2 .* routes.distance_miles

    pairs = []
    for morning in route_indexes, night in route_indexes
        morning == night && continue
        distance = round_trip_distance[morning] + round_trip_distance[night]
        distance <= max_daily_distance || continue
        run_count = routes.has_run_area[morning] + routes.has_run_area[night]
        run_count >= 1 || continue

        route_mask = (UInt64(1) << (morning - 1)) | (UInt64(1) << (night - 1))
        area_mask = area_bit[routes.area[morning]] | area_bit[routes.area[night]]
        score = weights.run * run_count - weights.distance * distance

        push!(pairs, (
            route_mask = route_mask,
            area_mask = area_mask,
            score = round(Int, score * scale),
        ))
    end

    # State is (yesterday's route mask, areas visited so far, score excluding area reward).
    states = Dict{Tuple{UInt64, UInt64, Int}, BigInt}((UInt64(0), UInt64(0), 0) => BigInt(1))

    for _ in DAYS
        next_states = Dict{Tuple{UInt64, UInt64, Int}, BigInt}()
        for ((previous_route_mask, visited_area_mask, score), count) in states
            for pair in pairs
                (previous_route_mask & pair.route_mask) == 0 || continue

                key = (
                    pair.route_mask,
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
            println("  ", rpad(row.walk_time * ":", 9), "$(row.route_id) ($(row.area), $(run_label), $(round(row.round_trip_distance_miles, digits = 2)) mi under foot)")
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
    routes = load_routes(options["routes"])
    weights = (
        area = parse(Float64, options["w-area"]),
        run = parse(Float64, options["w-run"]),
        distance = parse(Float64, options["w-distance"]),
    )

    schedule, objective, optimal_solution_count, raw_search_space_size = solve_schedule(
        routes;
        max_daily_distance = parse(Float64, options["max-daily-distance"]),
        weights,
    )
    print_schedule(schedule, objective, optimal_solution_count, raw_search_space_size)
end

main()
