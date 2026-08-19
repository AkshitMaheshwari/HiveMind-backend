-- =============================================================================
-- Supabase PostgreSQL Schema for Universal Multi-Agent Orchestrator
-- Auth + RBAC (User vs Admin) + Isolated Task History + Live Events
-- =============================================================================

-- 1. Profiles Table (Extends Supabase Auth users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT DEFAULT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for role lookup
CREATE INDEX IF NOT EXISTS idx_profiles_role ON public.profiles(role);

-- Automated trigger to create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, role)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', SPLIT_PART(NEW.email, '@', 1)),
        'user'
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- 2. Tasks Table (Linked to User ID)
CREATE TABLE IF NOT EXISTS public.tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL DEFAULT 'default',
    user_request TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued', -- 'queued', 'running', 'done', 'error'
    task_plan JSONB DEFAULT NULL,
    final_output TEXT DEFAULT NULL,
    error TEXT DEFAULT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ DEFAULT NULL
);

-- Ensure user_id column exists if table was created previously
ALTER TABLE public.tasks ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

-- Index for fast user query
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON public.tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON public.tasks(created_at DESC);


-- 3. Task Events Table (Streaming execution logs)
CREATE TABLE IF NOT EXISTS public.task_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES public.tasks(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    department TEXT DEFAULT NULL,
    agent TEXT DEFAULT NULL,
    data TEXT DEFAULT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_events_task_id ON public.task_events(task_id, id ASC);


-- 4. Helper Function to Prevent Infinite RLS Recursion
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM public.profiles
        WHERE id = auth.uid() AND role = 'admin'
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER STABLE;


-- 5. Row Level Security (RLS) Policies
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.task_events ENABLE ROW LEVEL SECURITY;

-- Drop existing policies to cleanly recreate
DROP POLICY IF EXISTS "Users can read own profile" ON public.profiles;
DROP POLICY IF EXISTS "Admins can read all profiles" ON public.profiles;
DROP POLICY IF EXISTS "Users can select own tasks" ON public.tasks;
DROP POLICY IF EXISTS "Users can insert own tasks" ON public.tasks;
DROP POLICY IF EXISTS "Users can update own tasks" ON public.tasks;
DROP POLICY IF EXISTS "Admins can read all tasks" ON public.tasks;
DROP POLICY IF EXISTS "Users can select task_events" ON public.task_events;
DROP POLICY IF EXISTS "Users can insert task_events" ON public.task_events;

-- Profiles: Users can read own profile; Admins can read all profiles
CREATE POLICY "Users can read own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id OR public.is_admin());

-- Tasks: Users can read & write ONLY their own tasks; Admins can read all
CREATE POLICY "Users can select own tasks" ON public.tasks
    FOR SELECT USING (auth.uid() = user_id OR public.is_admin());

CREATE POLICY "Users can insert own tasks" ON public.tasks
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own tasks" ON public.tasks
    FOR UPDATE USING (auth.uid() = user_id OR public.is_admin());

-- Task Events: Users can only view & insert events for their own tasks
CREATE POLICY "Users can select task_events" ON public.task_events
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.tasks
            WHERE public.tasks.id = public.task_events.task_id
            AND (public.tasks.user_id = auth.uid() OR public.is_admin())
        )
    );

CREATE POLICY "Users can insert task_events" ON public.task_events
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.tasks
            WHERE public.tasks.id = public.task_events.task_id
            AND (public.tasks.user_id = auth.uid() OR public.is_admin())
        )
    );


-- 6. Audit Logs Table (Optional persistent audit records)
CREATE TABLE IF NOT EXISTS public.audit_logs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    event_type TEXT NOT NULL,
    department TEXT DEFAULT NULL,
    agent TEXT DEFAULT NULL,
    data JSONB DEFAULT '{}'::jsonb
);

ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Admins can view audit_logs" ON public.audit_logs;
CREATE POLICY "Admins can view audit_logs" ON public.audit_logs
    FOR SELECT USING (public.is_admin());

DROP POLICY IF EXISTS "Allow service role or users to insert audit_logs" ON public.audit_logs;
CREATE POLICY "Allow service role or users to insert audit_logs" ON public.audit_logs
    FOR INSERT WITH CHECK (true);

