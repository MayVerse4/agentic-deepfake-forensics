// The Python backend (backend/forensics_api.py) is reached through the Next.js
// /backend proxy (see next.config.js), so requests are same-origin and the
// httpOnly session cookie is sent automatically.
export const API = '/backend';

export interface SessionUser {
  id: string;
  email: string;
  name: string;
}
