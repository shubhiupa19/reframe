# Journal History Feature - TODO

## Backend (Flask)
- [ ] Step 2: Add `/signup` endpoint (validate email, hash password with bcrypt, store user in DB)
- [ ] Step 2: Add `/login` endpoint (verify password, issue JWT, set httpOnly + Secure + SameSite=Lax cookie)
- [ ] Step 2: Add `/logout` endpoint (clear cookie)
- [ ] Step 3: Add JWT verification middleware in Flask to protect routes
- [ ] Step 3: Add `POST /entries` endpoint (save entry text + sentences for logged-in user)
- [ ] Step 3: Add `GET /entries` endpoint (return all entries + sentences for logged-in user)

## Frontend (Next.js)
- [ ] Step 4: Build `/login` page with email/password form
- [ ] Step 4: Build `/signup` page with email/password form
- [ ] Step 5: Add `middleware.js` at project root — check for auth cookie, redirect unauthenticated users away from protected pages
- [ ] Step 6: Build `/history` page — list past journal entries
- [ ] Step 6: Add distortion frequency chart (e.g. "your most common distortions this month")
- [ ] Step 6: Save entry to DB after analysis on the main page
