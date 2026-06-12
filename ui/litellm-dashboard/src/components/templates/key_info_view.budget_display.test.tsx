import { renderWithProviders } from "../../../tests/test-utils";
import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { KeyResponse } from "../key_team_helpers/key_list";
import KeyInfoView from "./key_info_view";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";

// IMPORTANT: do not mock `@/utils/dataUtils` here. We want to exercise the
// real `formatNumberWithCommas` so this test catches the LIT-2845 regression
// where the overview "Spend" card formatted `max_budget` with the default 0
// decimals, truncating sub-dollar budgets (e.g. $0.10) to "$0".

vi.mock("./key_edit_view", () => ({
  KeyEditView: () => <div data-testid="key-edit-view-stub" />,
}));

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({ default: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/projects/useProjects", () => ({
  useProjects: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/keys/useResetKeySpend", () => ({
  useResetKeySpend: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
}));
vi.mock("../networking", () => ({
  keyDeleteCall: vi.fn().mockResolvedValue({}),
  keyUpdateCall: vi.fn().mockResolvedValue({}),
  getPolicyInfoWithGuardrails: vi
    .fn()
    .mockResolvedValue({ resolved_guardrails: [] }),
}));

const MOCK_KEY_DATA = {
  token: "test-token-123",
  token_id: "test-token-123",
  key_name: "sk-...abcd",
  key_alias: "lit-2845-budget-display",
  spend: 0.0001,
  max_budget: null as number | null,
  expires: "null",
  models: [],
  aliases: {},
  config: {},
  user_id: "default_user_id",
  team_id: null,
  max_parallel_requests: null,
  metadata: {},
  tpm_limit: null,
  rpm_limit: null,
  budget_duration: null,
  budget_reset_at: null,
  allowed_cache_controls: [],
  permissions: {},
  model_spend: {},
  model_max_budget: {},
  soft_budget_cooldown: false,
  blocked: false,
  litellm_budget_table: {},
  organization_id: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  team_spend: 0,
  team_alias: "",
  team_tpm_limit: null,
  team_rpm_limit: null,
  team_max_budget: null,
  team_models: [],
  team_blocked: false,
  soft_budget: null,
  team_model_aliases: {},
  team_member_spend: 0,
  team_metadata: {},
  end_user_id: null,
  end_user_tpm_limit: null,
  end_user_rpm_limit: null,
  end_user_max_budget: null,
  last_refreshed_at: 0,
  api_key: "sk-...abcd",
  user_role: "user",
  rpm_limit_per_model: {},
  tpm_limit_per_model: {},
  user_tpm_limit: null,
  user_rpm_limit: null,
  user_email: "test@example.com",
  object_permission: {
    object_permission_id: "perm-1",
    mcp_servers: [],
    mcp_access_groups: [],
    mcp_tool_permissions: {},
    vector_stores: [],
  },
  auto_rotate: false,
  rotation_interval: undefined,
  last_rotation_at: undefined,
  key_rotation_at: undefined,
} as unknown as KeyResponse;

const baseAuthorized = {
  accessToken: "test-token",
  userId: "test-user",
  userRole: "admin",
  premiumUser: true,
  token: "test-token",
  userEmail: null,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
};

describe("KeyInfoView overview budget display (LIT-2845)", () => {
  beforeEach(() => {
    vi.mocked(useTeams).mockReturnValue({ teams: [], setTeams: vi.fn() });
    vi.mocked(useAuthorized).mockReturnValue(baseAuthorized);
  });

  it("renders a sub-dollar max_budget ($0.10) with 2-decimal precision in the overview Spend card", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: 0.1 }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    // Regression for LIT-2845: the overview card used to call
    // `formatNumberWithCommas(max_budget)` with no second arg, which
    // defaults to 0 decimals — so $0.10 rendered as "$0".
    // After the fix it must render "$0.10".
    await waitFor(() => {
      expect(screen.getByText(/of \$0\.10/)).toBeInTheDocument();
    });
    expect(screen.queryByText(/of \$0$/)).not.toBeInTheDocument();
  });

  it("still renders whole-dollar max_budget with 2-decimal precision", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: 100 }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await waitFor(() => {
      // 2-decimal formatting -> "$100.00"
      expect(screen.getByText(/of \$100\.00/)).toBeInTheDocument();
    });
  });

  it("renders 'Unlimited' when max_budget is null", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, max_budget: null }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(screen.getByText(/of Unlimited/)).toBeInTheDocument();
    });
  });

  it("renders a Budget Windows card with per-window spend, percent, and reset countdown when budget_limits are present", async () => {
    renderWithProviders(
      <KeyInfoView
        keyData={{
          ...MOCK_KEY_DATA,
          spend: 15,
          budget_limits: [
            {
              budget_duration: "5h",
              max_budget: 30,
              reset_at: "2026-06-12T20:00:00+00:00",
              reset_in_seconds: 191,
              reset_in: "3m 11s",
            },
            {
              budget_duration: "1w",
              max_budget: 60,
              reset_at: "2026-06-15T00:00:00+00:00",
              reset_in_seconds: 200000,
              reset_in: "2d 7h 33m 20s",
            },
            {
              budget_duration: "1mo",
              max_budget: 90,
              reset_at: "2026-07-01T00:00:00+00:00",
              reset_in_seconds: 1569791,
              reset_in: "18d 4h 3m 11s",
            },
          ],
        }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Budget Windows")).toBeInTheDocument();
    });

    expect(screen.getByText("5h")).toBeInTheDocument();
    expect(screen.getByText("1w")).toBeInTheDocument();
    expect(screen.getByText("1mo")).toBeInTheDocument();

    expect(screen.getByText(/\$15\.0000 of \$30\.00/)).toBeInTheDocument();
    expect(screen.getByText(/\$15\.0000 of \$60\.00/)).toBeInTheDocument();
    expect(screen.getByText(/\$15\.0000 of \$90\.00/)).toBeInTheDocument();

    expect(screen.getByText("50%")).toBeInTheDocument();
    expect(screen.getByText("25%")).toBeInTheDocument();
    expect(screen.getByText("16%")).toBeInTheDocument();

    expect(screen.getByText("Resets in 3m 11s")).toBeInTheDocument();
    expect(screen.getByText("Resets in 2d 7h 33m 20s")).toBeInTheDocument();
    expect(screen.getByText("Resets in 18d 4h 3m 11s")).toBeInTheDocument();

    expect(screen.getAllByTestId("budget-window-row")).toHaveLength(3);
  });

  it("does not render the Budget Windows card when budget_limits is undefined or empty", async () => {
    const { rerender } = renderWithProviders(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByText("Budget Windows")).not.toBeInTheDocument();
    });
    expect(screen.queryByTestId("budget-window-row")).not.toBeInTheDocument();

    rerender(
      <KeyInfoView
        keyData={{ ...MOCK_KEY_DATA, budget_limits: [] }}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByText("Budget Windows")).not.toBeInTheDocument();
    });
    expect(screen.queryByTestId("budget-window-row")).not.toBeInTheDocument();
  });
});
