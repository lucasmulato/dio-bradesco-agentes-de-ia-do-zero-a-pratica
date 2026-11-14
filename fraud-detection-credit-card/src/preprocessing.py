from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess(df):

    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    
    X = df.drop(columns=['Class', 'Amount', 'Time'])
    y = df['Class']
    
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
