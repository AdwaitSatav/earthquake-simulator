import numpy as np


def simulate_earthquakes(rate, time_period, magnitude_threshold=0.0, seed=None):
    """
    Simulate earthquake occurrences using a Poisson process.
    Returns list of (time, magnitude) tuples.
    """
    if seed is not None:
        np.random.seed(seed)

    events = []
    current_time = 0.0

    while current_time < time_period:
        interarrival = np.random.exponential(1.0 / rate)
        current_time += interarrival
        if current_time < time_period:
            magnitude = simulate_magnitude()
            if magnitude >= magnitude_threshold:
                events.append((current_time, magnitude))

    return events


def simulate_magnitude(a=8.0, b=1.0, m_min=2.0, m_max=9.5):
    """
    Simulate earthquake magnitude using the Gutenberg-Richter law.
    log10(N) = a - b*M  =>  M = (a - log10(U)) / b
    Uses inverse CDF sampling from truncated exponential.
    """
    beta = b * np.log(10)
    u = np.random.uniform(0, 1)
    magnitude = m_min - (1.0 / beta) * np.log(1 - u * (1 - np.exp(-beta * (m_max - m_min))))
    return np.clip(magnitude, m_min, m_max)


def get_interarrival_times(events):
    """Extract interarrival times from a list of (time, magnitude) event tuples."""
    if len(events) < 2:
        return np.array([])
    times = np.array([e[0] for e in events])
    return np.diff(times)


def monte_carlo_simulation(rate, time_period, n_simulations=10000, magnitude_threshold=0.0):
    """
    Run Monte Carlo simulations and return array of earthquake counts per simulation.
    """
    counts = []
    for _ in range(n_simulations):
        events = simulate_earthquakes(rate, time_period, magnitude_threshold)
        counts.append(len(events))
    return np.array(counts)


def simulate_with_coordinates(rate, time_period, magnitude_threshold=0.0,
                               lat_range=(-60, 70), lon_range=(-180, 180)):
    """
    Simulate earthquakes with random geographic coordinates.
    Returns list of dicts: {time, magnitude, lat, lon, depth}
    """
    events = simulate_earthquakes(rate, time_period, magnitude_threshold)
    geo_events = []
    for t, mag in events:
        geo_events.append({
            "time": t,
            "magnitude": mag,
            "lat": np.random.uniform(*lat_range),
            "lon": np.random.uniform(*lon_range),
            "depth": np.random.exponential(30),  # km, exponential depth distribution
        })
    return geo_events
