create table public.mcp_auth_codes (
  code text not null,
  client_id text not null,
  code_challenge text not null,
  redirect_uri text not null,
  user_id uuid not null,
  access_token text not null,
  refresh_token text not null,
  scope text null,
  used boolean null default false,
  expires_at timestamp with time zone not null,
  created_at timestamp with time zone null default now(),
  constraint mcp_auth_codes_pkey primary key (code),
  constraint mcp_auth_codes_user_id_fkey foreign KEY (user_id) references auth.users (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_mcp_auth_codes_expires on public.mcp_auth_codes using btree (expires_at) TABLESPACE pg_default;