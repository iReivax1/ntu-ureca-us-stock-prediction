# Project Installation Guide

#### 1. Install requirements
```
        $ pip install -r requirements.txt
```

#### 2. Setup database on postgres on the `postgres` console using the following command
```
# CREATE USER usnewsapp WITH PASSWORD 'qwe123qwe123';
# CREATE DATABASE usnews OWNER usnewsapp;
# ALTER ROLE usnewsapp superuser;
# ALTER ROLE usnewsapp SET client_encoding TO 'utf8';
# ALTER ROLE usnewsapp SET default_transaction_isolation TO 'read committed';
# ALTER ROLE usnewsapp SET timezone TO 'UTC';
# GRANT ALL PRIVILEGES ON DATABASE usnews TO usnewsapp;
```

#### 3. Run migrations to create the database schemas
```
        $ python manage.py makemigrations
        $ python manage.py migrate
```

#### 4. Rename environment variables file
rename `dot.env` to `.env` and fill in your twitter API keys
