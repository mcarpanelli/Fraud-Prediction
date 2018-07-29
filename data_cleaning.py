import pandas as pd
import numpy as np

#extracts data from 'ticket_types' column
def extract_ticket_type_col(ticket_type_col):
    ticket_df = pd.DataFrame(columns=['avg_ticket_price',
                                   'max_ticket_price',
                                   'ticket_tiers',
                                   'total_available'])
    for row in ticket_type_col:
        if row != []:
            num_tickets = sum([ticket.get('quantity_total') for ticket in row])
            max_price = max([ticket.get('cost', 0) for ticket in row])
            ticket_tiers = len(row)
            weighted_prices = sum([ticket.get('cost', 0) * ticket.get('quantity_total', 0)                                    for ticket in row])
            if num_tickets == 0:
                avg_price = np.mean([ticket.get('cost', 0) for ticket in row])
            else:
                avg_price = weighted_prices / num_tickets
            key_data = pd.Series({'avg_ticket_price': avg_price,
                                  'max_ticket_price': max_price,
                                  'ticket_tiers': ticket_tiers,
                                  'total_available': num_tickets})
        else:
            key_data = pd.Series({'avg_ticket_price': 0.,
                                  'max_ticket_price': 0.,
                                  'ticket_tiers': 0,
                                  'total_available': 0})
        ticket_df = ticket_df.append(key_data, ignore_index=True)
    ticket_df['ticket_tiers'] = ticket_df['ticket_tiers'].astype(int)
    ticket_df['total_available'] = ticket_df['ticket_tiers'].astype(int)
    return ticket_df



# calculates percentage of capital letters (aka SCREAMING) in string
def calc_screaming_factor(description):
    percent = np.array([])
    for row in description:
        caps = 0
        if row == '':
            percent_caps = 0
        else:
            for char in row:
                if char == char.upper():
                    caps += 1
            percent_caps = float(caps) / len(row)
        percent = np.append(percent, percent_caps)
    return percent

def clean_data(df):
    #create 'event_start_to_payout'
    df['event_start_to_payout'] = df['approx_payout_date'] - df['event_start']
    #create 'event_created_to_payout'
    df['event_created_to_payout'] = df['approx_payout_date'] - df['event_created']
    #create 'event_duration'
    df['event_duration'] = df['event_end'] - df['event_start']
    #create 'has_event_pub_date'
    df['event_published'].fillna(0, inplace=True)
    df['has_event_pub_date'] = df['event_published'] != 0
    df['has_event_pub_date'] = df['has_event_pub_date'].astype(int)

    #calculate percentage of caps in 'name' field
    df['screaming_factor'] = calc_screaming_factor(df['name'])

    # has_header fill missing values
    df['has_header'] = df[['has_header']].fillna(0.)

    # 'listed' map to boolean
    df['listed'] = df['listed'].map({'y':1, 'n':0})

    # add 'has_org_desc' boolean
    df['has_org_desc'] = (df['org_desc'] == '').astype(int)

    # create 'org_facebook_is_nan'
    df['org_facebook_nan'] = df['org_facebook'].isnull().map({True:1, False:0})

    # fill 'org_facebook' null values
    df['org_facebook'] = df['org_facebook'].fillna(0.)

    # add has_org_name
    df['has_org_name'] = (df['org_name'] == '').map({True:1, False:0})

    #add 'org_twitter_is_nan'
    df['org_twitter_is_nan'] = df['org_twitter'].isnull().map({True:1, False:0})

    # fillna 'org_twitter' null values
    df['org_twitter'] = df['org_twitter'].fillna(0.)

    # add 'payee_name_blank' bool
    df['payee_name_blank'] = (df['payee_name'] == '').map({True:1, False:0})

    #create 'ACH' and 'CHECK' cols
    dummies = pd.get_dummies(df['payout_type'], drop_first = True)
    df = pd.concat([df, dummies], axis=1)

    # create number of previous payouts
    df['num_previous_payouts'] = df['previous_payouts'].apply(len)

    # extract info from ticket_types
    df = pd.concat([df, extract_ticket_type_col(df['ticket_types'])], axis=1)

    # create 'country_is_venue_country' boolean if 'country' == 'venue_country'
    df['country_is_venue_country'] = 0
    mask = df['country']==df['venue_country']
    df.loc[mask,'country_is_venue_country'] = 1

    # create boolean for nan values of 'venue_name'
    df['venue_name_is_nan'] = 0
    df.loc[df['venue_name'].isnull(),'venue_name_is_nan'] = 1
    # fill nans with blank str
    df['venue_name'].fillna('')
    #create boolean for blank values of 'venue_name'
    df['venue_name_is_blank'] = 0
    mask = df['venue_name'] == ''
    df.loc[mask,'venue_name_is_blank'] = 1

    #drop  cols
    drop_cols = ['acct_type','country','currency','delivery_method',
    'email_domain', 'event_end', 'event_published', 'event_start',
    'name', 'org_desc', 'org_name', 'payee_name', 'payout_type',
    'user_created', 'sale_duration', 'previous_payouts', 'ticket_types',
    'user_type', 'venue_country', 'venue_latitude', 'venue_longitude',
    'venue_name', 'venue_state', 'approx_payout_date', 'description',
    'event_created', 'object_id', 'venue_address']
    df.drop(drop_cols, axis=1, inplace=True)

    return df

