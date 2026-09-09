"""cogutil's `Logger`, the logging surface every MOSES module writes through.

C++: opencog/util/Logger.h:51 Logger

Classic MOSES writes a single, line-oriented log file whose every record is

    [2026-09-09 18:24:03:734] [INFO] moses version "3"."7"."0"

-- a UTC timestamp (`gmtime_r`, NOT localtime), `strftime("%F %T")`, then a
**colon** and three-digit milliseconds, then the level in brackets, then the
message. The colon-before-milliseconds is not a typo in this docstring: the
format string is `"[%s:%03ld] "` at Logger.cc:512-513, so the separator
between seconds and milliseconds is the same character that separates hours
from minutes. A `.` there would be wrong.

## What is ported, and what is not

Ported: the `Level` enum and its string conversions (both directions, case
insensitive), the `logger()` singleton, level filtering, the timestamp /
level / component / thread-id prefixes and their flags, the file-vs-stdout
split, the lazy file open (Logger.cc:284-297), and the per-filename
`LogWriter` sharing (Logger.cc:405-421) that makes two `Logger` objects on
the same path append to one open handle.

# DIVERGENCE(C++ opencog/util/Logger.cc:196 Logger::LogWriter::writing_loop):
# classic hands each formatted record to a `concurrent_queue<std::string*>`
# drained by one dedicated writer thread, and `flush()` busy-waits on that
# queue. This port writes synchronously inside `log()`. The observable
# difference is nil for a single-threaded run -- classic `fflush`es after
# every record anyway (Logger.cc:311) and MOSES's own default is one
# thread -- and reproducing the thread would make the port's log ORDER
# depend on scheduling, which is exactly what the differential oracle
# compares.

# DIVERGENCE(C++ opencog/util/Logger.cc:74 prt_backtrace): classic is built
# with `-DHAVE_GNU_BACKTRACE` (CogUtil's CMakeLists.txt:128), so every
# record at or below `backTraceLevel` (default ERROR) gets a symbolized C
# stack appended. Python has no equivalent -- a Python traceback names
# different frames entirely -- so `backtrace()` is a no-op here and `log()`
# appends nothing. The flush-on-error behaviour it guards IS reproduced.
# Nothing in the differential corpus emits at ERROR, so this is invisible
# to the oracle; it is recorded because it would not be if one did.

# DIVERGENCE(C++ opencog/util/Logger.h:215 Logger::Base): the `logger().debug()
# << x << y` streaming form relies on a temporary's destructor firing at the
# end of the full expression to emit the accumulated line. Python has no
# such guarantee, so this port exposes only the string / printf-style form,
# `logger().debug(msg)` and `logger().debug(fmt, *args)`. Call sites that
# stream in C++ build the same string with an f-string instead. The five
# `LAZY_LOG_*` macros (Logger.h:388-392) collapse into ordinary
# `if logger().is_*_enabled():` guards for the same reason.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from enum import IntEnum

__all__ = ["Logger", "logger", "ostream_double"]

# C++: Logger.cc:70 `const char* levelStrings[]`. Indexed BY the enum value,
# which is why NONE must stay at index 0.
_LEVEL_STRINGS = ("NONE", "ERROR", "WARN", "INFO", "DEBUG", "FINE")

# C++: Logger.cc:69 MAX_PRINTF_STYLE_MESSAGE_SIZE -- `vsnprintf` into a
# 32768-byte stack buffer, so a longer printf-style message is TRUNCATED
# rather than wrapped or split. Reproduced in `logva`.
_MAX_PRINTF_STYLE_MESSAGE_SIZE = 1 << 15


class _LogWriter:
    """C++: opencog/util/Logger.h:339 Logger::LogWriter

    One writer per file name, shared between every `Logger` pointed at that
    name (`Logger::set_filename`, Logger.cc:405-421, looks the writer up in a
    static map before constructing one). The file handle is opened lazily, on
    the first record actually written -- Logger.cc:284-286's own comment: "this
    allows us to set the main logger's filename without creating a useless log
    file with the default filename". That laziness is observable: a run that
    logs nothing leaves no file behind, and `problem_params` relies on it,
    since the singleton is constructed pointing at `opencog.log` long before
    `-f` is parsed.
    """

    def __init__(self) -> None:
        self._file_name = ""
        self._logfile = None

    def set_file_name(self, s: str) -> None:
        """C++: opencog/util/Logger.cc:390 Logger::LogWriter::setFileName"""
        self._file_name = s
        if self._logfile is not None:
            self._logfile.flush()
            self._logfile.close()
        self._logfile = None

    def get_file_name(self) -> str:
        """C++: opencog/util/Logger.h:365 Logger::LogWriter::getFileName"""
        return self._file_name

    def qmsg(self, msg: str) -> None:
        """C++: opencog/util/Logger.h:368 Logger::LogWriter::qmsg

        C++ pushes onto `msg_queue`; the writer thread pops and calls
        `write_msg`. This port calls `write_msg` directly -- see the module
        docstring's threading DIVERGENCE.
        """
        self._write_msg(msg)

    def size(self) -> int:
        """C++: opencog/util/Logger.h:371 Logger::LogWriter::size

        Always 0: nothing is ever queued (see `qmsg`). Kept because
        `Logger.log` reads it to decide whether to force a flush, and dropping
        the read would drop a branch classic really has.
        """
        return 0

    def flush(self) -> None:
        """C++: opencog/util/Logger.cc:261 Logger::LogWriter::flush

        C++ busy-waits for the queue to drain and then `fdatasync`s. Here the
        queue is always empty, so only the sync remains.
        """
        if self._logfile is not None:
            self._logfile.flush()
            os.fsync(self._logfile.fileno())

    def _write_msg(self, msg: str) -> None:
        """C++: opencog/util/Logger.cc:280 Logger::LogWriter::write_msg

        `fopen(fileName, "a")` -- append, not truncate. Classic's
        `problem_params::parse_options` calls `remove(log_file)` first
        (problem-params.cc:2091) precisely because this never truncates.
        """
        if self._logfile is None:
            try:
                self._logfile = open(self._file_name, "a", encoding="utf-8", newline="\n")
            except OSError:
                sys.stderr.write(f'[ERROR] Unable to open log file "{self._file_name}"\n')
                return
        self._logfile.write(msg)
        # C++: Logger.cc:309-319 -- "log messages are important, especially if
        # we are about to crash", so every record is flushed immediately.
        self._logfile.flush()


# C++: Logger.cc:145-146 `std::mutex Logger::_loggers_mtx; std::map<std::string,
# Logger::LogWriter*> Logger::_loggers;` -- class statics, so module-level here.
# No mutex: see the module docstring's threading DIVERGENCE.
_LOGGERS: dict[str, _LogWriter] = {}


class Logger:
    """C++: opencog/util/Logger.h:51 Logger"""

    class Level(IntEnum):
        """C++: opencog/util/Logger.h:56 Logger::Level

        `enum Level { NONE, ERROR, WARN, INFO, DEBUG, FINE, BAD_LEVEL=255 };`
        The ORDER is load-bearing twice over: `Logger::log` filters with
        `level > currentLevel`, so a numerically larger level is a *more*
        verbose one, and `get_level_from_string` returns the index into
        `levelStrings` directly (Logger.cc:626-635), so the enum and the string
        table must stay in step.
        """

        NONE = 0
        ERROR = 1
        WARN = 2
        INFO = 3
        DEBUG = 4
        FINE = 5
        BAD_LEVEL = 255

    @staticmethod
    def get_level_from_string(level_str: str) -> Logger.Level:
        """C++: opencog/util/Logger.cc:626 Logger::get_level_from_string

        `strcasecmp` against each entry of `levelStrings`, returning the index;
        `BAD_LEVEL` when nothing matches. Case-insensitive, and note it does
        NOT trim -- classic's caller (problem-params.cc:2093) applies
        `boost::trim` itself before calling this.
        """
        for i, name in enumerate(_LEVEL_STRINGS):
            if level_str.casefold() == name.casefold():
                return Logger.Level(i)
        return Logger.Level.BAD_LEVEL

    @staticmethod
    def get_level_string(level: Logger.Level) -> str:
        """C++: opencog/util/Logger.cc:618 Logger::get_level_string

        `BAD_LEVEL` prints as "Bad level"; every other value indexes
        `levelStrings`. C++ indexes unchecked, so a value outside 0..5 is
        undefined behaviour there rather than an error -- this port raises
        instead of reading out of bounds.
        """
        if level == Logger.Level.BAD_LEVEL:
            return "Bad level"
        return _LEVEL_STRINGS[int(level)]

    def __init__(
        self,
        file_name: str = "opencog.log",
        level: Logger.Level = Level.INFO,
        timestamp_enabled: bool = True,
    ) -> None:
        """C++: opencog/util/Logger.cc:322 Logger

        Defaults come from the declaration (Logger.h:76-77): file
        "opencog.log", level INFO, timestamps on. The body then sets
        `backTraceLevel = ERROR`, `threadIdEnabled = false`,
        `printLevel = true`, `printToStdout = false`, `syncEnabled = false`
        and `logEnabled = true`.
        """
        self._component = ""
        self._log_writer: _LogWriter | None = None
        self.set_filename(file_name)

        self._current_level = level
        self._back_trace_level = Logger.Level.ERROR

        self._timestamp_enabled = timestamp_enabled
        self._thread_id_enabled = False
        self._print_level = True
        self._print_to_stdout = False
        self._sync_enabled = False

        self._log_enabled = True

    def set_level(self, level: Logger.Level | str) -> None:
        """C++: opencog/util/Logger.cc:370 Logger::set_level

        Merges the two C++ overloads: `set_level(Level)` (Logger.cc:370) and
        the inline `set_level(const std::string&)` (Logger.h:92-94), which is
        just `set_level(get_level_from_string(str))`.
        """
        if isinstance(level, str):
            level = Logger.get_level_from_string(level)
        self._current_level = level

    def get_level(self) -> Logger.Level:
        """C++: opencog/util/Logger.cc:375 Logger::get_level"""
        return self._current_level

    def set_backtrace_level(self, level: Logger.Level | str) -> None:
        """C++: opencog/util/Logger.cc:380 Logger::set_backtrace_level

        Merges the `Level` overload with the inline string one (Logger.h:109).
        """
        if isinstance(level, str):
            level = Logger.get_level_from_string(level)
        self._back_trace_level = level

    def get_backtrace_level(self) -> Logger.Level:
        """C++: opencog/util/Logger.cc:385 Logger::get_backtrace_level"""
        return self._back_trace_level

    def set_filename(self, fname: str) -> None:
        """C++: opencog/util/Logger.cc:405 Logger::set_filename

        Reuses an existing `LogWriter` for `fname` if one exists -- classic's
        own comment: "other logger instances might also be using this writer!"
        -- and calls `enable()` unconditionally at the end.
        """
        writer = _LOGGERS.get(fname)
        if writer is None:
            writer = _LogWriter()
            writer.set_file_name(fname)
            _LOGGERS[fname] = writer
        self._log_writer = writer
        self._enable()

    def get_filename(self) -> str:
        """C++: opencog/util/Logger.cc:423 Logger::get_filename

        Returns an empty string when there is no writer, matching C++'s
        `static std::string bad;`.
        """
        if self._log_writer is None:
            return ""
        return self._log_writer.get_file_name()

    def set_component(self, c: str) -> None:
        """C++: opencog/util/Logger.cc:430 Logger::set_component"""
        self._component = c

    def get_component(self) -> str:
        """C++: opencog/util/Logger.cc:435 Logger::get_component"""
        return self._component

    def set_timestamp_flag(self, flag: bool) -> None:
        """C++: opencog/util/Logger.cc:440 Logger::set_timestamp_flag"""
        self._timestamp_enabled = flag

    def set_thread_id_flag(self, flag: bool) -> None:
        """C++: opencog/util/Logger.cc:445 Logger::set_thread_id_flag"""
        self._thread_id_enabled = flag

    def get_thread_id_flag(self) -> bool:
        """C++: opencog/util/Logger.cc:450 Logger::get_thread_id_flag"""
        return self._thread_id_enabled

    def set_print_to_stdout_flag(self, flag: bool) -> None:
        """C++: opencog/util/Logger.cc:455 Logger::set_print_to_stdout_flag"""
        self._print_to_stdout = flag

    def get_print_to_stdout_flag(self) -> bool:
        """C++: opencog/util/Logger.cc:460 Logger::get_print_to_stdout_flag"""
        return self._print_to_stdout

    def set_print_level_flag(self, flag: bool) -> None:
        """C++: opencog/util/Logger.cc:465 Logger::set_print_level_flag"""
        self._print_level = flag

    def set_sync_flag(self, flag: bool) -> None:
        """C++: opencog/util/Logger.cc:470 Logger::set_sync_flag"""
        self._sync_enabled = flag

    def set_print_error_level_stdout(self) -> None:
        """C++: opencog/util/Logger.cc:475 Logger::set_print_error_level_stdout"""
        self.set_print_to_stdout_flag(True)
        self.set_level(Logger.Level.ERROR)

    def _enable(self) -> None:
        """C++: opencog/util/Logger.cc:481 Logger::enable"""
        self._log_enabled = True

    def _disable(self) -> None:
        """C++: opencog/util/Logger.cc:486 Logger::disable"""
        self._log_enabled = False

    def log(self, level: Logger.Level, txt: str, *args: object) -> None:
        """C++: opencog/util/Logger.cc:491 Logger::log

        Merges C++'s two `log` overloads: `log(Level, const std::string&)`
        (Logger.cc:491) and the printf-style `log(Level, const char*, ...)`
        (Logger.cc:588), which forwards to `logva`. Passing extra arguments
        selects the second: it formats through `logva` first, INCLUDING that
        overload's own extra `level <= currentLevel` pre-check and its
        truncation at 32768 bytes.

        The record is assembled exactly as Logger.cc:499-526 does:
        timestamp, then level, then component, then thread id, then the
        message, then a newline. Each prefix is followed by a single space and
        is omitted entirely when its flag is off (an empty component prints no
        brackets at all -- Logger.cc:520).
        """
        if args:
            self.logva(level, txt, args)
            return

        # C++: Logger.cc:495-497. Three separate early returns, in this order.
        if not self._log_enabled:
            return
        if level > self._current_level:
            return
        if self._log_writer is None:
            return

        oss = []
        if self._timestamp_enabled:
            oss.append(f"[{_timestamp()}] ")

        if self._print_level:
            oss.append(f"[{Logger.get_level_string(level)}] ")

        if self._component:
            oss.append(f"[{self._component}] ")

        if self._thread_id_enabled:
            oss.append(f"[thread-{_thread_id()}] ")

        # C++: `oss << txt << std::endl;` -- std::endl is '\n' plus a stream
        # flush, and the flush is a no-op on an ostringstream.
        oss.append(txt)
        oss.append("\n")
        msg = "".join(oss)

        self._log_writer.qmsg(msg)

        # C++: Logger.cc:537-540. `size()` is identically 0 here, so this
        # branch never fires -- see `_LogWriter.size`.
        if self._log_writer.size() > 1024:
            self.flush()

        # C++: Logger.cc:542-545 -- errors precede crashes, so get the record
        # onto the disk before the crash can happen.
        if level <= self._back_trace_level:
            self.flush()

        if self._sync_enabled:
            self.flush()

        if self._print_to_stdout:
            sys.stdout.write(msg)
            sys.stdout.flush()

    def logva(self, level: Logger.Level, fmt: str, args: tuple[object, ...]) -> None:
        """C++: opencog/util/Logger.cc:578 Logger::logva

        Note the redundant-looking `if (level <= currentLevel)` here: `log`
        checks the same thing again. It is not redundant in effect -- it stops
        `vsnprintf` running at all for a filtered message, which is why a
        printf-style call is cheap when the level is off and a streamed one is
        not.

        `vsnprintf` into a 32768-byte buffer truncates rather than growing;
        reproduced by slicing. C++ counts BYTES and reserves one for the NUL
        terminator, so the cut is at 32767 bytes of encoded output.
        """
        if level <= self._current_level:
            buffer = fmt % args
            encoded = buffer.encode("utf-8", errors="replace")
            if len(encoded) > _MAX_PRINTF_STYLE_MESSAGE_SIZE - 1:
                encoded = encoded[: _MAX_PRINTF_STYLE_MESSAGE_SIZE - 1]
                buffer = encoded.decode("utf-8", errors="ignore")
            self.log(level, buffer)

    def backtrace(self) -> None:
        """C++: opencog/util/Logger.cc:557 Logger::backtrace

        A no-op beyond the null-writer guard -- see the module docstring's
        backtrace DIVERGENCE. C++ queues the formatted stack; with nothing to
        format, the queued string would be empty.
        """
        if self._log_writer is None:
            return

    def error(self, txt: str, *args: object) -> None:
        """C++: opencog/util/Logger.h:238 Logger::Error

        `logger().error(txt)` is `Logger::Error::operator()(const
        std::string&)` (Logger.h:241); `logger().error(fmt, ...)` is
        `Logger::Error::operator()(const char*, ...)` (Logger.cc:593), which
        routes through `logva`. The streaming `logger().error()` form is not
        ported -- see the module docstring.
        """
        self.log(Logger.Level.ERROR, txt, *args)

    def warn(self, txt: str, *args: object) -> None:
        """C++: opencog/util/Logger.h:250 Logger::Warn

        See `error` -- the real symbols are `Logger::Warn::operator()`
        (Logger.h:253 and Logger.cc:598).
        """
        self.log(Logger.Level.WARN, txt, *args)

    def info(self, txt: str, *args: object) -> None:
        """C++: opencog/util/Logger.h:262 Logger::Info

        See `error` -- the real symbols are `Logger::Info::operator()`
        (Logger.h:265 and Logger.cc:603).
        """
        self.log(Logger.Level.INFO, txt, *args)

    def debug(self, txt: str, *args: object) -> None:
        """C++: opencog/util/Logger.h:274 Logger::Debug

        See `error` -- the real symbols are `Logger::Debug::operator()`
        (Logger.h:277 and Logger.cc:608).
        """
        self.log(Logger.Level.DEBUG, txt, *args)

    def fine(self, txt: str, *args: object) -> None:
        """C++: opencog/util/Logger.h:286 Logger::Fine

        See `error` -- the real symbols are `Logger::Fine::operator()`
        (Logger.h:289 and Logger.cc:613).
        """
        self.log(Logger.Level.FINE, txt, *args)

    def is_enabled(self, level: Logger.Level) -> bool:
        """C++: opencog/util/Logger.h:305 Logger::is_enabled"""
        return level <= self._current_level

    def is_error_enabled(self) -> bool:
        """C++: opencog/util/Logger.h:306 Logger::is_error_enabled"""
        return Logger.Level.ERROR <= self._current_level

    def is_warn_enabled(self) -> bool:
        """C++: opencog/util/Logger.h:307 Logger::is_warn_enabled"""
        return Logger.Level.WARN <= self._current_level

    def is_info_enabled(self) -> bool:
        """C++: opencog/util/Logger.h:308 Logger::is_info_enabled"""
        return Logger.Level.INFO <= self._current_level

    def is_debug_enabled(self) -> bool:
        """C++: opencog/util/Logger.h:309 Logger::is_debug_enabled"""
        return Logger.Level.DEBUG <= self._current_level

    def is_fine_enabled(self) -> bool:
        """C++: opencog/util/Logger.h:310 Logger::is_fine_enabled"""
        return Logger.Level.FINE <= self._current_level

    def flush(self) -> None:
        """C++: opencog/util/Logger.cc:256 Logger::flush"""
        if self._log_writer is not None:
            self._log_writer.flush()


def ostream_double(x: float) -> str:
    """C++: opencog/util/Logger.h:215 Logger::Base

    Not a C++ symbol of its own: `logger().debug() << some_double` writes
    through a freshly constructed `std::stringstream` (`Logger::Base::ss`,
    Logger.h:234), which carries an `ostream`'s DEFAULT precision of 6 and
    `defaultfloat` formatting -- equivalent to C's `%.6g`, which Python's
    `".6g"` format spec matches byte for byte (the same equivalence
    `iostream_combo._contin_to_str` already relies on, verified there
    against `combo-fmt-converter` over 3.14, 3.0, 0.0001, 1e-100, -0.0 and
    others).

    This is NOT `io_score_precision`. That 18-digit setting is applied by
    `composite_score`'s own `operator<<` via an explicit `setprecision`;
    a log record that never calls `setprecision` does not inherit it,
    because each record gets its own fresh stream. A `score_t` streamed
    bare into a log line therefore prints at SIX digits even though the
    same value inside a printed `composite_score` prints at eighteen --
    both appear in classic's log, two lines apart.

    A NEGATIVE NaN prints as `-nan`, not `nan`. libstdc++ and glibc both
    honour NaN's sign bit -- measured on this toolchain 2026-09-09:
    `std::cout << -std::nan("")` writes `-nan` at default precision and at
    18 -- while Python's `format` writes plain `nan` for either sign.
    `0.0f/0.0f` yields a NEGATIVE NaN on x86-64, so this is the ordinary
    case: classic's own `-Hprerec` log writes `Score: -nan` for a candidate
    that selects no rows.
    """
    if x != x:  # NaN
        return "-nan" if math.copysign(1.0, x) < 0.0 else "nan"
    return format(x, ".6g")


def _timestamp() -> str:
    """C++: opencog/util/Logger.cc:500 Logger::log (the `timestampEnabled` block)

    Logger.cc:502-514, verbatim:

        ::gettimeofday(&stv, NULL);
        time_t t = stv.tv_sec;
        gmtime_r(&t, &stm);
        strftime(timestamp, sizeof(timestamp), "%F %T", &stm);
        snprintf(timestampStr, sizeof(timestampStr),
                "[%s:%03ld] ", timestamp, (long)stv.tv_usec / 1000);

    Three details decide whether a line matches classic:

    * `gmtime_r`, so the stamp is UTC, not local time.
    * `%F %T` is `%Y-%m-%d %H:%M:%S` -- ISO date, space, 24-hour clock.
    * the milliseconds are joined with a COLON, not a dot, and come from
      integer-truncating microseconds (`tv_usec / 1000`), not from rounding.

    This returns the inside of the brackets; `Logger.log` adds them, exactly
    as the C++ format string does.
    """
    now = time.time()
    # `gettimeofday` yields whole seconds plus whole microseconds; take both
    # from the same reading so the two halves cannot straddle a second.
    secs = int(now)
    usec = int((now - secs) * 1_000_000)
    stm = time.gmtime(secs)
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', stm)}:{usec // 1000:03d}"


def _thread_id() -> object:
    """C++: opencog/util/Logger.cc:523 Logger::log (the `threadIdEnabled` block)

    `oss << "[thread-" << std::this_thread::get_id() << "] "`. libstdc++
    streams a `thread::id` as the underlying `pthread_t`, a large opaque
    integer; Python's `threading.get_ident()` is the same kind of opaque
    per-thread token. The two never agree numerically. Nothing in MOSES sets
    `threadIdEnabled` (grep: zero calls to `set_thread_id_flag` under
    $CLASSIC/moses/), so no differential run can observe the difference.
    """
    return threading.get_ident()


# C++: Logger.cc:638-642 `Logger& opencog::logger() { static Logger instance;
# return instance; }` -- a function-local static, so it is constructed on the
# first call and lives until process exit. A module-level `None` plus the
# lazy construction in `logger()` reproduces both halves.
_INSTANCE: Logger | None = None


def logger() -> Logger:
    """C++: opencog/util/Logger.cc:638 logger

    The process-wide singleton every MOSES translation unit writes through.
    Constructed on first use with the `Logger()` defaults -- file
    "opencog.log", level INFO, timestamps on -- which is why `-l`/`-f` can be
    applied by `problem_params::parse_options` long after the first call: the
    log FILE is not opened until the first record is actually written.
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = Logger()
    return _INSTANCE
