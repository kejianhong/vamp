#pragma once

#include <array>
#include <cstdio>
#include <type_traits>
#include <utility>

#include <vamp/planning/constraints/manifold/constraint.hh>
#include <vamp/planning/constraints/utils.hh>
#include <vamp/vector.hh>

namespace vamp::planning::constraint
{
    // Whether a robot's generated header provides the bounded least-squares TSR step,
    // Robot::solve_tsr_error_bounded_ls. The kernel is emitted by cricket for robots that were
    // (re)generated after ProjMethod::BoundedLS was introduced; headers from before that have
    // no such member, so BoundedLS has to fall back to InnerLM there rather than fail to
    // compile. Probed by substitution on the call expression: a missing member is a
    // substitution failure, an existing one is instantiated once and then reused by the
    // actual call.
    template <typename Robot, std::size_t rake, typename Input, typename = void>
    struct has_bounded_tsr_step : std::false_type
    {
    };

    template <typename Robot, std::size_t rake, typename Input>
    struct has_bounded_tsr_step<
        Robot,
        rake,
        Input,
        std::void_t<decltype(Robot::template solve_tsr_error_bounded_ls<rake>(
            std::declval<const Input &>(),
            std::declval<typename Robot::template ConfigurationBlock<rake> &>()))>> : std::true_type
    {
    };

    // Task Space Region constraint (Berenson et al., "Task Space Regions", IJRR 2011): for
    // each end-effector, the pose of an offset frame e (rTe, expressed in the end-effector
    // frame) must lie within [lower, upper] se(3) bounds of a reference frame r (wTr,
    // expressed in the world frame). Transforms are wxyz quaternion + xyz translation,
    // matching the input layout of the generated Robot::tsr_error.
    template <typename Robot, std::size_t rake>
    struct TaskSpaceConstraint final
      : HingedTSRConstraint<TaskSpaceConstraint<Robot, rake>, Robot, rake, 6 * Robot::n_eef>
    {
        using Base =
            HingedTSRConstraint<TaskSpaceConstraint<Robot, rake>, Robot, rake, 6 * Robot::n_eef>;
        using Block = typename Base::Block;
        using Row = typename Base::Row;

        static constexpr std::size_t n_eef = Robot::n_eef;

        // Same shape as the TSR solve buffer, plus the per-dimension step limits.
        using BoundedInput =
            BoundedSolveInput<Robot, rake, 6 * n_eef, 6 * n_eef * Robot::dimension>;

        // This constraint has a generated bounded least-squares step (ProjMethod::BoundedLS)
        // when the robot header provides it; see has_bounded_tsr_step above. False keeps the
        // class compiling against older robot headers, where step() sends BoundedLS to InnerLM.
        static constexpr bool bounded_ls = has_bounded_tsr_step<Robot, rake, BoundedInput>::value;

        using Transform = std::array<float, 7>;
        using Bound = std::array<float, 6>;

        TaskSpaceConstraint(
            const std::array<Transform, n_eef> &eef_to_offset,       // rTe per end-effector
            const std::array<Transform, n_eef> &world_to_reference,  // wTr per end-effector
            const std::array<Bound, n_eef> &lower,
            const std::array<Bound, n_eef> &upper) noexcept
        {
            for (auto i = 0U; i < n_eef; ++i)
            {
                for (auto j = 0U; j < 7; ++j)
                {
                    input.rTe[i * 7 + j] = Row::fill(eef_to_offset[i][j]);
                    input.wTr[i * 7 + j] = Row::fill(world_to_reference[i][j]);
                }

                for (auto j = 0U; j < 6; ++j)
                {
                    input.lb[i * 6 + j] = Row::fill(lower[i][j]);
                    input.ub[i * 6 + j] = Row::fill(upper[i][j]);
                    this->equality_rows[i * 6 + j] = (upper[i][j] - lower[i][j]) < equality_row_width;
                }
            }
        }

    private:
        friend Base;

        void run_kernel() const noexcept
        {
            Robot::template tsr_error<rake>(input, this->solve);
        }

        // The step of ProjMethod::BoundedLS: `input` is a BoundedSolveInput (solve buffer + step limits).
        // Only instantiated when the robot header provides the kernel (bounded_ls); step() guards
        // the call with if constexpr, so robots without it never name this function.
        template <typename Input>
        void solve_step_bounded(Block &gradient, const Input &input) const noexcept
        {
            static_assert(
                bounded_ls,
                "this robot's generated header has no solve_tsr_error_bounded_ls kernel; "
                "regenerate it with a cricket that emits ProjMethod::BoundedLS");
            Robot::template solve_tsr_error_bounded_ls<rake>(input, gradient);
        }

        void solve_step(Block &gradient, ProjMethod method) const noexcept
        {
            switch (method)
            {
                case ProjMethod::BoundedLS:
                    if constexpr (not bounded_ls)
                    {
                        // step() only dispatches BoundedLS to solve_step_bounded when the
                        // robot has the kernel, so reaching this branch means the header is
                        // older than the method: say so instead of silently running InnerLM.
                        warn_bounded_ls_fallback();
                    }
                    [[fallthrough]];
                case ProjMethod::InnerLM:
                    Robot::template solve_tsr_error_lm_inner<rake>(this->solve, gradient);
                    break;
                case ProjMethod::OuterLM:
                    Robot::template solve_tsr_error_lm_outer<rake>(this->solve, gradient);
                    break;
                case ProjMethod::GradDesc:
                    Robot::template solve_tsr_error_gradient_descent<rake>(this->solve, gradient);
                    break;
            }
        }

        // Once per process (per robot/rake instantiation): tell the user that BoundedLS is not
        // available for this robot instead of silently planning with InnerLM.
        static void warn_bounded_ls_fallback() noexcept
        {
            static bool warned = false;
            if (warned)
            {
                return;
            }

            warned = true;
            std::fprintf(
                stderr,
                "vamp: ProjMethod::BoundedLS is not available for this robot (its generated "
                "header has no solve_tsr_error_bounded_ls kernel); falling back to InnerLM. "
                "Regenerate the robot header with a cricket that emits the bounded solver to "
                "use BoundedLS.\n");
        }

        // Input layout of the generated tsr_error: q, then per end-effector rTe (7), wTr (7),
        // lower (6), upper (6).
        struct Input
        {
            Block q;
            FloatVector<rake, 7 * n_eef> rTe;
            FloatVector<rake, 7 * n_eef> wTr;
            FloatVector<rake, 6 * n_eef> lb;
            FloatVector<rake, 6 * n_eef> ub;

            static constexpr std::size_t per_eef = 7 + 7 + 6 + 6;

            auto operator[](std::size_t index) const noexcept -> Row
            {
                if (index < Robot::dimension)
                {
                    return q[index];
                }

                const std::size_t eef = (index - Robot::dimension) / per_eef;
                const std::size_t i = (index - Robot::dimension) % per_eef;

                if (i < 7)
                {
                    return rTe[eef * 7 + i];
                }

                if (i < 14)
                {
                    return wTr[eef * 7 + i - 7];
                }

                if (i < 20)
                {
                    return lb[eef * 6 + i - 14];
                }

                return ub[eef * 6 + i - 20];
            }
        };

        mutable Input input;
    };
}  // namespace vamp::planning::constraint
