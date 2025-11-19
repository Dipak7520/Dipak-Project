import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Currency conversion rate (EUR to INR)
# As of November 2025, 1 EUR ≈ 90 INR (approximate rate)
EUR_TO_INR_RATE = 90.0

def generate_hostel_data(n_samples=1000, price_multiplier=1.0):
    """
    Generate synthetic hostel data for training models
    
    Args:
        n_samples (int): Number of samples to generate
        price_multiplier (float): Factor to multiply all prices by
        
    Returns:
        pd.DataFrame: Generated hostel data
    """
    # Define cities and their base prices (higher prices in more expensive cities)
    # Added Indian cities with appropriate base prices
    cities = {
        'Paris': 45, 'London': 40, 'Amsterdam': 42, 'Berlin': 35, 'Barcelona': 38,
        'Rome': 36, 'Prague': 30, 'Vienna': 37, 'Budapest': 28, 'Dublin': 41,
        'Madrid': 34, 'Athens': 32, 'Lisbon': 33, 'Warsaw': 29, 'Stockholm': 48,
        'Copenhagen': 47, 'Oslo': 50, 'Helsinki': 44, 'Brussels': 43, 'Milan': 46,
        # Indian cities with appropriate base prices (converted to EUR)
        'Mumbai': 25, 'Delhi': 22, 'Bangalore': 24, 'Hyderabad': 23, 'Chennai': 21,
        'Kolkata': 20, 'Pune': 23, 'Ahmedabad': 22, 'Jaipur': 24, 'Goa': 30,
        'Kochi': 26, 'Mysore': 25, 'Darjeeling': 27, 'Shimla': 28, 'Manali': 29
    }
    
    # Initialize lists to store data
    data = {
        'city': [],
        'distance_to_center_km': [],
        'rating': [],
        'num_reviews': [],
        'room_type': [],
        'beds_in_room': [],
        'breakfast_included': [],
        'wifi': [],
        'laundry': [],
        'season': [],
        'weekend': [],
        'occupancy_rate': [],
        'hostel_age_years': [],
        'kitchen_access': [],
        'air_conditioning': [],
        'locker_available': [],
        'security_24h': [],
        'female_only_dorm_available': [],
        'common_area_count': [],
        'room_area_sqft': [],
        'bathroom_type': [],
        'price_per_night': []
    }
    
    # Generate samples
    for _ in range(n_samples):
        # Select a random city
        city = random.choice(list(cities.keys()))
        base_price = cities[city]
        
        # Distance to center (0.1 to 8 km)
        distance_to_center = round(random.uniform(0.1, 8.0), 1)
        
        # Rating (2.5 to 5.0)
        rating = round(random.uniform(2.5, 5.0), 1)
        
        # Number of reviews (based on rating - higher rated hostels have more reviews)
        if rating >= 4.5:
            num_reviews = random.randint(150, 500)
        elif rating >= 4.0:
            num_reviews = random.randint(80, 300)
        elif rating >= 3.5:
            num_reviews = random.randint(30, 150)
        else:
            num_reviews = random.randint(5, 100)
        
        # Room type
        room_type = random.choice(['dorm', 'private'])
        
        # Beds in room (more for dorms)
        if room_type == 'dorm':
            beds_in_room = random.choice([4, 6, 8, 10, 12])
        else:
            beds_in_room = random.choice([1, 2])
        
        # Amenities (higher probability for better rated hostels)
        amenity_prob = min(0.9, rating / 5.0)
        breakfast_included = int(random.random() < amenity_prob * 0.8)
        wifi = 1  # Almost always available
        laundry = int(random.random() < amenity_prob * 0.7)
        
        # Season (affects pricing)
        season = random.choice(['low', 'shoulder', 'peak'])
        
        # Weekend (affects pricing)
        weekend = random.choice([0, 1])
        
        # Occupancy rate (higher in peak season)
        if season == 'peak':
            occupancy_rate = round(random.uniform(0.7, 1.0), 2)
        elif season == 'shoulder':
            occupancy_rate = round(random.uniform(0.5, 0.8), 2)
        else:
            occupancy_rate = round(random.uniform(0.3, 0.7), 2)
        
        # Hostel age (0 to 30 years)
        hostel_age_years = random.randint(0, 30)
        
        # More amenities for newer hostels
        age_factor = max(0.5, 1 - hostel_age_years / 60)
        kitchen_access = int(random.random() < amenity_prob * age_factor)
        air_conditioning = int(random.random() < amenity_prob * age_factor * 0.8)
        locker_available = int(random.random() < amenity_prob * 0.9)
        security_24h = int(random.random() < amenity_prob * 0.85)
        female_only_dorm_available = int(random.random() < 0.3)
        
        # Common areas (1 to 6)
        common_area_count = random.randint(1, 6)
        
        # Room area (depends on room type)
        if room_type == 'dorm':
            room_area_sqft = random.randint(120, 250)
        else:
            room_area_sqft = random.randint(100, 200)
        
        # Bathroom type
        bathroom_type = random.choice(['shared', 'private'])
        
        # Calculate price based on all factors
        # Base price adjusted by city
        price = base_price
        
        # Adjust by distance (farther is cheaper)
        price *= max(0.5, 1 - distance_to_center / 20)  # Ensure minimum 0.5 multiplier
        
        # Adjust by rating (higher rating = higher price)
        price *= (0.8 + rating / 5 * 0.4)
        
        # Adjust by room type (private is more expensive)
        if room_type == 'private':
            price *= 1.8
        
        # Adjust by season
        if season == 'peak':
            price *= 1.4
        elif season == 'shoulder':
            price *= 1.1
        
        # Adjust by weekend
        if weekend:
            price *= 1.2
        
        # Adjust by occupancy (higher occupancy = higher price)
        price *= (0.8 + occupancy_rate * 0.5)
        
        # Adjust by hostel age (older hostels are cheaper)
        price *= max(0.5, 1 - hostel_age_years / 100)  # Ensure minimum 0.5 multiplier
        
        # Adjust by amenities
        amenity_bonus = (breakfast_included + kitchen_access + air_conditioning + 
                        locker_available + security_24h) * 2
        price += amenity_bonus
        
        # Adjust by beds in room (more beds = cheaper per bed)
        price /= max(1.0, beds_in_room ** 0.3)  # Ensure minimum 1.0 divisor
        
        # Add some noise
        price *= random.uniform(0.85, 1.15)
        
        # Apply price multiplier
        price *= price_multiplier
        
        # Ensure minimum price of 10
        price = max(10, price)
        
        # Store data
        data['city'].append(city)
        data['distance_to_center_km'].append(distance_to_center)
        data['rating'].append(rating)
        data['num_reviews'].append(num_reviews)
        data['room_type'].append(room_type)
        data['beds_in_room'].append(beds_in_room)
        data['breakfast_included'].append(breakfast_included)
        data['wifi'].append(wifi)
        data['laundry'].append(laundry)
        data['season'].append(season)
        data['weekend'].append(weekend)
        data['occupancy_rate'].append(occupancy_rate)
        data['hostel_age_years'].append(hostel_age_years)
        data['kitchen_access'].append(kitchen_access)
        data['air_conditioning'].append(air_conditioning)
        data['locker_available'].append(locker_available)
        data['security_24h'].append(security_24h)
        data['female_only_dorm_available'].append(female_only_dorm_available)
        data['common_area_count'].append(common_area_count)
        data['room_area_sqft'].append(room_area_sqft)
        data['bathroom_type'].append(bathroom_type)
        data['price_per_night'].append(round(price, 2))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Clamp any negative values to zero (shouldn't happen with current logic, but just in case)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].clip(lower=0)
    
    return df

def convert_prices_to_inr(df, rate=EUR_TO_INR_RATE):
    """
    Convert prices in the dataframe from EUR to INR
    
    Args:
        df (pd.DataFrame): DataFrame with price_per_night column
        rate (float): Conversion rate from EUR to INR
        
    Returns:
        pd.DataFrame: DataFrame with prices converted to INR
    """
    df_copy = df.copy()
    df_copy['price_per_night_inr'] = df_copy['price_per_night'] * rate
    return df_copy

def save_data(df, filename='../data/hostel_data_extended.csv'):
    """
    Save generated data to CSV
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Filename to save to
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Generate extended dataset with price multiplier
    print("Generating synthetic hostel data...")
    df = generate_hostel_data(2000, price_multiplier=1.0)  # Default multiplier
    print(f"Generated {len(df)} samples")
    print(df.head())
    
    # Save to CSV
    save_data(df)
    
    # Display some statistics
    print("\nDataset Statistics:")
    print(f"Price range: €{df['price_per_night'].min():.2f} - €{df['price_per_night'].max():.2f}")
    print(f"Average price: €{df['price_per_night'].mean():.2f}")
    print(f"Price median: €{df['price_per_night'].median():.2f}")
    
    # Show INR conversion example
    sample_price_eur = df['price_per_night'].iloc[0]
    sample_price_inr = sample_price_eur * EUR_TO_INR_RATE
    print(f"\nSample price conversion: €{sample_price_eur:.2f} = ₹{sample_price_inr:.2f}")