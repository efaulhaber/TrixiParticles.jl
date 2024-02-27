using Interpolations
using Statistics

function calculate_mse(reference_time, reference_values, simulation_time, simulation_values)
    interp_func = LinearInterpolation(simulation_time, simulation_values,
                                      extrapolation_bc=Flat())

    # Find the common time range
    common_time_range = filter(t -> t >= maximum([
                                                minimum(simulation_time),
                                                minimum(reference_time),
                                            ]) &&
                                   t <= minimum([
                                               maximum(simulation_time),
                                               maximum(reference_time),
                                           ]), reference_time)

    # Interpolate simulation data at the common time points
    interpolated_values = [interp_func(t) for t in common_time_range]

    # Extract the corresponding reference displacement values
    filtered_values = [reference_values[findfirst(==(t),
                                                  reference_time)]
                       for t in common_time_range]

    # Calculate MSE only over the common time range
    mse = mean((interpolated_values .- filtered_values) .^ 2)
    return mse
end

function extract_number(filename)
    # This regex matches the last sequence of digits in the filename
    m = match(r"(\d+)(?!.*\d)", filename)
    if m !== nothing
        num = parse(Int, m.captures[1])
        return num
    else
        println("No numeric sequence found in filename: ", filename)
        return -1
    end
end

function find_and_compare_values(ref_data, run_data, errors=[])
    if isa(ref_data, Dict) && isa(run_data, Dict)
        for key in keys(ref_data)
            if key == "values" && haskey(run_data, key)
                ref_values = ref_data[key]
                run_values = run_data[key]
                if isa(ref_values, Array) && isa(run_values, Array)
                    for (ref_value, run_value) in zip(ref_values, run_values)
                        if isa(ref_value, Number) && isa(run_value, Number)
                            push!(errors, (ref_value - run_value)^2)
                        else
                            println("Non-numeric data encountered under 'values' key.")
                        end
                    end
                end
            elseif haskey(run_data, key)
                # Recursively search for "values" keys in nested dictionaries
                find_and_compare_values(ref_data[key], run_data[key], errors)
            end
        end
    end
    return errors
end

function calculate_error(ref_data, run_data)
    return sum(find_and_compare_values(ref_data, run_data))
end

function convert_to_float(str)
    # Remove leading zeros and count them
    leading_zeros = length(match(r"^0*", str).match)
    str_non_zero = replace(str, r"^0*" => "")

    if isempty(str_non_zero)
        return 0.0
    end

    # Adjust string to have a decimal point at the correct position
    if leading_zeros > 0
        decimal_str = "0." * "0"^(leading_zeros - 1) * str_non_zero
    else
        decimal_str = str_non_zero
    end

    return string(parse(Float64, decimal_str))
end