<think>
The user wants to know how to programmatically create a new user in macOS, similar to the `adduser` command in Linux.

 In macOS, there are several ways to create users programmatically:

 1. **Using `dscl` (Directory Service command line)** - This is the most common and traditional way, similar to manipulating the user database directly.

 2. **Using `sysadminctl`** - This is a newer tool (introduced in macOS 10.10+) specifically designed for user administration, and it's the recommended modern approach.

 3. **Using `createhomedir`** - For creating home directories after the user exists.

 4. **Using Objective-C/Swift with OpenDirectory frameworks** - For native app development.

 5. **Using AppleScript** - Though less common for this task.

 The most straightforward command-line approaches are:
 - `sysadminctl` (modern, easier)
 - `dscl` (lower level, more control)

 Let me provide examples for both, with explanations.

 Key considerations:
 - Need root/admin privileges (sudo)
 - Need to specify UID (or let system assign)
 - Need to create home directory
 - Need to set password
 - Groups membership (staff, admin, etc.)

 `sysadminctl` syntax:
 ```
 sudo sysadminctl -addUser <username> -fullName "<Full Name>" -password <password> [-admin] [-UID <uid>] [-shell <shell>] [-home <home_dir>]
 ```

 `dscl` approach involves multiple steps:
 1. Create the user record
 2. Set properties (UID, GID, shell, home, etc.)
 3. Create home directory
 4. Set password

 I should provide both methods, explain when to use each, and mention security considerations (passwords in scripts, etc.).

 Also worth mentioning:
 - The user must have a UniqueID (UID) - usually 501+ for regular users
 - PrimaryGroupID is usually 20 (staff) or 80 (admin)
 - Shell is usually /bin/bash or /bin/zsh (depending on macOS version)

 Let me structure the answer:
 1. Modern method (sysadminctl) - recommended
 2. Traditional method (dscl) - for more control
 3. Creating home directory
 4. Programmatic examples (bash scripts)
 5. Security warnings

 Actually, looking at the latest macOS versions, `sysadminctl` is definitely the way to go for simple user creation, while `dscl` gives more granular control.

 Let me verify the exact syntax for sysadminctl:
 `sudo sysadminctl -addUser username -fullName "Full Name" -password password`

 And for dscl, it's more complex:
 ```
 sudo dscl . -create /Users/username
 sudo dscl . -create /Users/username UserShell /bin/zsh
 sudo dscl . -create /Users/username RealName "Full Name"
 sudo dscl . -create /Users/username UniqueID 1001
 sudo dscl . -create /Users/username PrimaryGroupID 20
 sudo dscl . -create /Users/username NFSHomeDirectory /Users/username
 sudo dscl . -passwd /Users/username password
 ```

 Also mention `createhomedir -c -u username` to create the home directory properly.

 I should also mention that on macOS, users are stored in the local directory service (OpenDirectory), not /etc/passwd like traditional Unix.

 Let's provide a comprehensive answer.
</think>

There are two primary command-line methods to create users programmatically on macOS: **`sysadminctl`** (modern, recommended) and **`dscl`** (low-level, traditional).

## Method 1: sysadminctl (Recommended)

Introduced in macOS 10.10+, this is the cleanest approach:

```bash
sudo sysadminctl -addUser username \
    -fullName "Full Name" \
    -password "password" \
    -UID 1001 \
    -shell /bin/zsh \
    -home /Users/username \
    -admin
```

**Key flags:**
- `-admin`: Adds user to admin group (optional)
- `-UID`: Specify user ID (auto-assigned if omitted)
- `-shell`: Default shell (`/bin/zsh` on modern macOS, `/bin/bash` on older)
- `-home`: Home directory path

**Create home directory separately:**
```bash
sudo createhomedir -c -u username
```

## Method 2: dscl (Directory Service)

For finer control or older systems, use the Directory Service command line:

```bash
USERNAME="newuser"
FULLNAME="New User"
PASSWORD="temppass"
UID="1001"  # Must be unique, check with `id 1001`

# Create user record
sudo dscl . -create /Users/$USERNAME

# Set properties
sudo dscl . -create /Users/$USERNAME UserShell /bin/zsh
sudo dscl . -create /Users/$USERNAME RealName "$FULLNAME"
sudo dscl . -create /Users/$USERNAME UniqueID $UID
sudo dscl . -create /Users/$USERNAME PrimaryGroupID 20  # 20=staff, 80=admin
sudo dscl . -create /Users/$USERNAME NFSHomeDirectory /Users/$USERNAME

# Set password
sudo dscl . -passwd /Users/$USERNAME "$PASSWORD"

# Create home directory
sudo createhomedir -c -u $USERNAME
```

## Finding the next available UID

To avoid UID conflicts:

```bash
# Find highest existing UID and add 1
LASTUID=$(dscl . -list /Users UniqueID | awk '{print $2}' | sort -n | tail -1)
NEWUID=$((LASTUID + 1))
```

## Security Considerations

**⚠️ Warning about passwords:**
- Never hardcode passwords in scripts
- Use stdin or secure prompts:
  ```bash
  read -s PASSWORD
  sudo sysadminctl -addUser testuser -password $PASSWORD
  ```
- Or create user without password and set later:
  ```bash
  sudo dscl . -create /Users/username Password '*'
  # User cannot login until password is set
  ```

## Swift/Objective-C (Programmatic API)

For native apps, use OpenDirectory framework:

```swift
import OpenDirectory

let session = ODSession.default()
let node = try ODNode(session: session, type: .localNodes)
let user = try ODRecord(node: node, recordType: .users, name: "newuser", attributes: [
    kODAttributeTypeFullName: ["Full Name"],
    kODAttributeTypeUserShell: ["/bin/zsh"],
    kODAttributeTypeUniqueID: ["1001"],
    kODAttributeTypePrimaryGroupID: ["20"]
])
try user.changePassword(nil, toPassword: "newpassword")
```

## Verification

Check if user was created:
```bash
id username
dscl . -read /Users/username
```

**Note:** Unlike Linux's `/etc/passwd`, macOS uses the **OpenDirectory** database (accessible via `dscl`). Changes take effect immediately without needing to restart services.
