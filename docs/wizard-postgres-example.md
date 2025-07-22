# PostgreSQL Password Auto-Generation in Wizard

The Semantik setup wizard now includes automatic PostgreSQL password generation, providing the same level of security and convenience as JWT key generation.

## How It Works

When you run `make wizard`, the setup process now includes a PostgreSQL configuration step:

```
🚀 Semantik Setup Wizard
========================

✅ Python 3.12.x detected
✅ Poetry is installed
✅ Dependencies already installed

🧙 Starting interactive setup wizard...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2 of 5: Database Configuration

Database Configuration

Semantik uses PostgreSQL for storing user data, collections, and metadata.

? PostgreSQL Password: 
  ❯ Generate secure password automatically (Recommended)
    Enter custom password
```

## Generated Password Example

If you select automatic generation, the wizard creates a secure 32-byte hex password:

```python
import secrets
postgres_password = secrets.token_hex(32)
# Example: "a7f3b2c9d8e1f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x2y3z4a5b6c7d8e9"
```

## Configuration Review

The wizard shows the masked password in the review step:

```
Review Configuration

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting                ┃ Value                           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Database              │ PostgreSQL                       │
│ PostgreSQL Password   │ ***c7d8e9                       │
│ JWT Secret           │ ***a2b3c4                       │
└───────────────────────┴─────────────────────────────────┘
```

## Environment File

The password is automatically saved to `.env`:

```env
# PostgreSQL Configuration
POSTGRES_PASSWORD=a7f3b2c9d8e1f4g5h6i7j8k9l0m1n2o3p4q5r6s7t8u9v0w1x2y3z4a5b6c7d8e9
POSTGRES_DB=semantik
POSTGRES_USER=semantik
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
```

## Docker Compose Integration

The wizard automatically includes `docker-compose.postgres.yml` when building and starting services:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
```

## Security Features

1. **32-byte hex password**: Same strength as JWT keys (256 bits of entropy)
2. **Minimum length validation**: Custom passwords must be at least 16 characters
3. **Masked display**: Only last 8 characters shown in review
4. **Automatic backup**: Existing `.env` is backed up before changes
5. **Port checking**: Verifies PostgreSQL port 5432 is available

## Manual Alternative

If you prefer not to use the wizard, you can still generate a secure password manually:

```bash
# Using make docker-postgres-up (also auto-generates if needed)
make docker-postgres-up

# Or manually with openssl
openssl rand -hex 32
```

Then add to your `.env` file:

```env
POSTGRES_PASSWORD=your_generated_password_here
```

## Benefits

- **Zero configuration**: Users don't need to think about database passwords
- **Secure by default**: Cryptographically secure passwords
- **Consistent experience**: Same UX as JWT key generation
- **Reduced errors**: No weak or default passwords
- **Easy setup**: One command configures everything