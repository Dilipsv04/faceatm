import os
import cv2
import json
import uuid
import base64
import numpy as np
from io import BytesIO
from deepface import DeepFace
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Create database directory if it doesn't exist
DATABASE_PATH = "atm_db"
USERS_FILE = os.path.join(DATABASE_PATH, "users.json")
ACCOUNTS_FILE = os.path.join(DATABASE_PATH, "accounts.json")

if not os.path.exists(DATABASE_PATH):
    os.makedirs(DATABASE_PATH)

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)

if not os.path.exists(ACCOUNTS_FILE):
    with open(ACCOUNTS_FILE, 'w') as f:
        json.dump({}, f)

# Face authentication settings
MODEL_NAME = "VGG-Face"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.4  # Lower threshold = stricter matching

def base64_to_image(base64_string):
    """Convert base64 string to numpy array image."""
    # Remove data:image/jpeg;base64, prefix if present
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    # Decode base64 string to numpy array
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def get_users():
    """Get all registered users."""
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    return users

def get_accounts():
    """Get all account information."""
    with open(ACCOUNTS_FILE, 'r') as f:
        accounts = json.load(f)
    return accounts

def save_accounts(accounts):
    """Save accounts to file."""
    with open(ACCOUNTS_FILE, 'w') as f:
        json.dump(accounts, f, indent=4)

def verify_pin(username, pin):
    """Verify if the PIN is correct for the user. Uses a default PIN if no PIN is set."""
    accounts = get_accounts()
    default_pin = "1234"  # Default PIN fallback

    if username in accounts:
        stored_pin = accounts[username].get('pin', default_pin)  # Use default if missing
        return pin == stored_pin
    return False

def record_transaction(username, transaction_type, amount, recipient=None):
    """Record a transaction in the user's account."""
    accounts = get_accounts()
    
    if username not in accounts:
        return False
    
    transaction = {
        'type': transaction_type,
        'amount': amount,
        'timestamp': datetime.now().isoformat()
    }
    
    if recipient:
        transaction['recipient'] = recipient
    
    accounts[username]['transactions'].append(transaction)
    save_accounts(accounts)
    return True

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    username = request.form.get('username')
    face_data = request.form.get('face_data')
    
    # Input validation
    if not username or not face_data:
        flash('Username and face capture are required!', 'danger')
        return redirect(url_for('register'))
    
    # Load existing users
    users = get_users()
    
    # Check if username already exists
    if username in users:
        flash(f"User '{username}' already exists!", 'danger')
        return redirect(url_for('register'))
    
    try:
        # Convert base64 to image
        face_img = base64_to_image(face_data)
        
        # Verify that a face is detectable
        DeepFace.detectFace(face_img)
        
        # Generate a unique ID for the face image
        face_id = str(uuid.uuid4())
        face_path = os.path.join(DATABASE_PATH, f"{face_id}.jpg")
        
        # Save the face image
        cv2.imwrite(face_path, face_img)
        
        # Update the users file
        users[username] = {
            'face_id': face_id,
            'created_at': datetime.now().isoformat()
        }
        
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
        
        flash(f"User '{username}' registered successfully! Please create your PIN.", 'success')
        
        # After registration, redirect to create PIN
        session['temp_username'] = username
        return redirect(url_for('create_pin'))
            
    except Exception as e:
        flash(f"Error during registration: {str(e)}", 'danger')
        return redirect(url_for('register'))

@app.route('/create_pin', methods=['GET', 'POST'])
def create_pin():
    if 'temp_username' not in session:
        flash('Please register first!', 'warning')
        return redirect(url_for('register'))
    
    if request.method == 'GET':
        return render_template('create_pin.html')
    
    pin = request.form.get('pin')
    confirm_pin = request.form.get('confirm_pin')
    
    # Input validation
    if not pin or not confirm_pin:
        flash('PIN and confirmation are required!', 'danger')
        return redirect(url_for('create_pin'))
    
    if pin != confirm_pin:
        flash('PINs do not match!', 'danger')
        return redirect(url_for('create_pin'))
    
    if len(pin) != 4 or not pin.isdigit():
        flash('PIN must be a 4-digit number!', 'danger')
        return redirect(url_for('create_pin'))
    
    username = session['temp_username']
    
    # Load existing accounts
    accounts = get_accounts()
    
    # Create new account
    accounts[username] = {
        'pin': pin,
        'balance': 0.0,
        'transactions': [],
        'created_at': datetime.now().isoformat()
    }
    
    save_accounts(accounts)
    
    # Clean up session
    session.pop('temp_username')
    
    flash('PIN created successfully! You can now login.', 'success')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    username = request.form.get('username')
    face_data = request.form.get('face_data')
    
    # Input validation
    if not username or not face_data:
        flash('Username and face capture are required!', 'danger')
        return redirect(url_for('login'))
    
    # Load existing users
    users = get_users()
    
    # Check if username exists
    if username not in users:
        flash(f"User '{username}' does not exist!", 'danger')
        return redirect(url_for('login'))
    
    # Get the face ID for the user
    face_id = users[username]['face_id']
    reference_img_path = os.path.join(DATABASE_PATH, f"{face_id}.jpg")
    
    if not os.path.exists(reference_img_path):
        flash(f"Reference image for user '{username}' not found!", 'danger')
        return redirect(url_for('login'))
    
    try:
        # Convert base64 to image
        face_img = base64_to_image(face_data)
        
        # Save the login attempt temporarily for verification
        temp_img_path = os.path.join(DATABASE_PATH, "temp_login.jpg")
        cv2.imwrite(temp_img_path, face_img)
        
        # Verify the face
        result = DeepFace.verify(
            img1_path=temp_img_path,
            img2_path=reference_img_path,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC
        )
        
        # Clean up temporary file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        if result["verified"]:
            session['username'] = username
            flash(f"Face authentication successful! Welcome, {username}!", 'success')
            return redirect(url_for('verify_pin_route'))
        else:
            flash(f"Login failed! Face does not match for user '{username}'.", 'danger')
            return redirect(url_for('login'))
                
    except Exception as e:
        flash(f"Error during login: {str(e)}", 'danger')
        return redirect(url_for('login'))

@app.route('/verify_pin', methods=['GET', 'POST'])
def verify_pin_route():
    if 'username' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('verify_pin.html')
    
    pin = request.form.get('pin')
    
    # Input validation
    if not pin:
        flash('PIN is required!', 'danger')
        return redirect(url_for('verify_pin_route'))
    
    username = session['username']
    
    if verify_pin(username, pin):
        session['pin_verified'] = True
        flash('PIN verified successfully!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Incorrect PIN!', 'danger')
        return redirect(url_for('verify_pin_route'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))
    
    if 'pin_verified' not in session:
        flash('Please verify your PIN!', 'warning')
        return redirect(url_for('verify_pin_route'))
    
    username = session['username']
    accounts = get_accounts()
    
    if username not in accounts:
        flash('Account not found!', 'danger')
        return redirect(url_for('logout'))
    
    account = accounts[username]
    
    return render_template('dashboard.html', 
                           username=username, 
                           balance=account['balance'],
                           transactions=account['transactions'][-5:])  # Show last 5 transactions

@app.route('/withdraw', methods=['GET', 'POST'])
def withdraw():
    if 'username' not in session or 'pin_verified' not in session:
        flash('Authentication required!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('withdraw.html')
    
    amount = request.form.get('amount')
    pin = request.form.get('pin')
    
    # Input validation
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError("Amount must be positive")
    except ValueError:
        flash('Please enter a valid amount!', 'danger')
        return redirect(url_for('withdraw'))
    
    username = session['username']
    
    if not verify_pin(username, pin):
        flash('Incorrect PIN!', 'danger')
        return redirect(url_for('withdraw'))
    
    accounts = get_accounts()
    
    if username not in accounts:
        flash('Account not found!', 'danger')
        return redirect(url_for('dashboard'))
    
    if accounts[username]['balance'] < amount:
        flash('Insufficient funds!', 'danger')
        return redirect(url_for('withdraw'))
    
    # Process withdrawal
    accounts[username]['balance'] -= amount
    save_accounts(accounts)
    
    # Record transaction
    record_transaction(username, 'withdraw', amount)
    
    flash(f'Successfully withdraw ₹{amount:.2f}!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/deposit', methods=['GET', 'POST'])
def deposit():
    if 'username' not in session or 'pin_verified' not in session:
        flash('Authentication required!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('deposit.html')
    
    amount = request.form.get('amount')
    pin = request.form.get('pin')
    
    # Input validation
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError("Amount must be positive")
    except ValueError:
        flash('Please enter a valid amount!', 'danger')
        return redirect(url_for('deposit'))
    
    username = session['username']
    
    if not verify_pin(username, pin):
        flash('Incorrect PIN!', 'danger')
        return redirect(url_for('deposit'))
    
    accounts = get_accounts()
    
    if username not in accounts:
        flash('Account not found!', 'danger')
        return redirect(url_for('dashboard'))
    
    # Process deposit
    accounts[username]['balance'] += amount
    save_accounts(accounts)
    
    # Record transaction
    record_transaction(username, 'deposit', amount)
    
    flash(f'Successfully deposited ₹{amount:.2f}!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/send', methods=['GET', 'POST'])
def send():
    if 'username' not in session or 'pin_verified' not in session:
        flash('Authentication required!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('send.html')
    
    recipient = request.form.get('recipient')
    amount = request.form.get('amount')
    pin = request.form.get('pin')
    
    # Input validation
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError("Amount must be positive")
    except ValueError:
        flash('Please enter a valid amount!', 'danger')
        return redirect(url_for('send'))
    
    username = session['username']
    
    if not verify_pin(username, pin):
        flash('Incorrect PIN!', 'danger')
        return redirect(url_for('send'))
    
    accounts = get_accounts()
    
    if username not in accounts:
        flash('Account not found!', 'danger')
        return redirect(url_for('dashboard'))
    
    if recipient not in accounts:
        flash('Recipient not found!', 'danger')
        return redirect(url_for('send'))
    
    if accounts[username]['balance'] < amount:
        flash('Insufficient funds!', 'danger')
        return redirect(url_for('send'))
    
    # Process transfer
    accounts[username]['balance'] -= amount
    accounts[recipient]['balance'] += amount
    save_accounts(accounts)
    
    # Record transactions
    record_transaction(username, 'send', amount, recipient)
    record_transaction(recipient, 'receive', amount, username)
    
    flash(f'Successfully sent ₹{amount:.2f} to {recipient}!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/check_balance', methods=['GET', 'POST'])
def check_balance():
    if 'username' not in session or 'pin_verified' not in session:
        flash('Authentication required!', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'GET':
        return render_template('check_balance.html')
    
    pin = request.form.get('pin')
    username = session['username']
    
    if not verify_pin(username, pin):
        flash('Incorrect PIN!', 'danger')
        return redirect(url_for('check_balance'))
    
    accounts = get_accounts()
    
    if username not in accounts:
        flash('Account not found!', 'danger')
        return redirect(url_for('dashboard'))
    
    balance = accounts[username]['balance']
    
    return render_template('balance_display.html', balance=balance)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('pin_verified', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)