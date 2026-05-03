create table public.mcp_oauth_clients (
  client_id text not null,
  redirect_uris jsonb not null default '[]'::jsonb,
  client_name text not null default 'MCP Client'::text,
  grant_types jsonb null default '["authorization_code", "refresh_token"]'::jsonb,
  response_types jsonb null default '["code"]'::jsonb,
  token_endpoint_auth_method text null default 'none'::text,
  created_at timestamp with time zone null default now(),
  constraint mcp_oauth_clients_pkey primary key (client_id)
) TABLESPACE pg_default;